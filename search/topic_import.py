"""Persistent import jobs with a separate local worker, requiring no task broker."""
import os
import subprocess
import sys
import tempfile
import time
from xml.etree import ElementTree as ET
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.db import close_old_connections, transaction
from django.utils import timezone

from .forms import get_or_create_topic
from .indexer import DuplicateDocumentError, find_duplicate_document, index_file, parse_document
from .models import ImportJob
from .pmc_client import fetch_pubmed_batch, search_pubmed, NCBITransientError
from .db_retry import retry_sqlite_write
from .arxiv_client import search_arxiv


@retry_sqlite_write
def recover_stale_jobs():
    cutoff = timezone.now() - timedelta(minutes=10)
    stale = ImportJob.objects.filter(status__in=['queued', 'running'], updated_at__lt=cutoff)
    # Status polling must not acquire a write lock on every refresh.
    if not stale.exists():
        return 0
    return stale.update(
        status='interrupted', message='The worker stopped responding. Collected articles were kept. Start another import to continue.',
        updated_at=timezone.now(),
    )


@retry_sqlite_write
def create_job(query, count, source='pubmed'):
    if not 1 <= count <= 1000:
        raise ValueError('Request between 1 and 1,000 articles.')
    if source not in {'pubmed','arxiv'}:
        raise ValueError('Unknown article source.')
    recover_stale_jobs()
    with transaction.atomic():
        topic = get_or_create_topic(query)
        return ImportJob.objects.create(topic=topic, requested=count, source=source)


def launch_job(job):
    logs = Path(settings.BASE_DIR) / 'data' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['BIOMEDIR_DB'] = str(settings.DATABASES['default']['NAME'])
    kwargs = {}
    if os.name == 'nt':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs['start_new_session'] = True
    try:
        with (logs / f'import-{job.pk}.log').open('ab') as log:
            subprocess.Popen(
                [sys.executable, str(Path(settings.BASE_DIR) / 'manage.py'), 'run_topic_import', str(job.pk)],
                cwd=str(settings.BASE_DIR), env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, **kwargs,
            )
    except OSError:
        job.status = 'failed'
        job.message = 'The import worker could not start. Please restart the app and try again.'
        job.save()


@retry_sqlite_write
def _persist(job, **fields):
    for name, value in fields.items():
        setattr(job, name, value)
    # Never overwrite a stop request written by the web process.
    job.save(update_fields=[*fields, 'imported', 'linked', 'duplicates',
                            'skipped', 'examined', 'offset', 'available', 'updated_at'])


def _cancelled(job):
    close_old_connections()
    if ImportJob.objects.get(pk=job.pk).cancel_requested:
        _persist(job, status='cancelled', message='Import stopped. All collected articles have been kept.')
        return True
    return False


class ImportCancelled(Exception):
    pass


def _source_call(job, operation, *args, **kwargs):
    # _get already retries an individual request. A longer, visible cooldown
    # lets a large import survive temporary source failures between batches.
    for attempt in range(4):
        try:
            return operation(*args, **kwargs)
        except NCBITransientError as exc:
            if attempt == 3:
                raise
            seconds = (3, 8, 20)[attempt]
            _persist(job, message=f'Temporary {job.get_source_display()} connection issue. Retrying in {seconds}s ({attempt+1}/3); {job.added} articles are saved. {exc}')
            for _ in range(seconds):
                if _cancelled(job):
                    raise ImportCancelled()
                time.sleep(1)
            _persist(job, message=f'Retrying {job.get_source_display()}; previously collected articles remain saved.')


@retry_sqlite_write
def _claim_job(job_id):
    return ImportJob.objects.filter(pk=job_id, status='queued').update(status='running', updated_at=timezone.now())


@retry_sqlite_write
def _store_record(job, pmid, data, title, meta, corpus):
    """Commit one article, its topic, index and counts together, or retry all."""
    before = (job.imported, job.linked, job.duplicates)
    target = None
    try:
        with transaction.atomic():
            existing = find_duplicate_document(title, meta, '')
            if existing:
                if existing.topics.filter(pk=job.topic_id).exists():
                    job.duplicates += 1
                else:
                    existing.topics.add(job.topic)
                    job.linked += 1
            else:
                prefix='arXiv' if job.source=='arxiv' else 'PMID'
                safe_id=pmid.replace('/','_')
                target = corpus / f'{prefix}{safe_id}_{job.pk.hex[:8]}.xml'
                target.write_bytes(data)
                try:
                    with transaction.atomic():
                        doc = index_file(str(target))
                        doc.topics.add(job.topic)
                    job.imported += 1
                except DuplicateDocumentError as exc:
                    target.unlink(missing_ok=True)
                    target = None
                    if exc.document.topics.filter(pk=job.topic_id).exists():
                        job.duplicates += 1
                    else:
                        exc.document.topics.add(job.topic)
                        job.linked += 1
            _persist(job, message=f'Adding to {job.topic.name} · {job.added:,} / {job.requested:,}')
        target = None  # Committed XML belongs to the collection now.
    except Exception:
        if target:
            target.unlink(missing_ok=True)
        job.imported, job.linked, job.duplicates = before
        raise


def run_job(job_id):
    close_old_connections()
    # Only one process may claim this queued job.
    claimed = _claim_job(job_id)
    if not claimed:
        return
    job = ImportJob.objects.select_related('topic').get(pk=job_id)
    corpus = Path(settings.BASE_DIR) / 'data' / 'corpus'
    corpus.mkdir(parents=True, exist_ok=True)
    try:
        while job.added < job.requested:
            if _cancelled(job):
                return
            if job.offset >= 10000:
                break
            _persist(job, message=f'Finding relevant abstracts on {job.get_source_display()}…')
            arxiv_records=[]
            if job.source=='arxiv':
                arxiv_records,count=_source_call(job,search_arxiv,job.topic.name,retstart=job.offset,retmax=min(100,10000-job.offset))
                ids=[identifier for identifier,_ in arxiv_records]
            else:
                ids, count = _source_call(job, search_pubmed, job.topic.name, retstart=job.offset, retmax=min(200, 10000-job.offset))
            _persist(job, available=count)
            if not ids:
                break
            for start in range(0, len(ids), 50):
                if _cancelled(job):
                    return
                batch = ids[start:start+50]
                _persist(job, message=f'Downloading abstracts · {job.added:,} / {job.requested:,} added')
                records = arxiv_records[start:start+50] if job.source=='arxiv' else _source_call(job, fetch_pubmed_batch, batch)
                returned = {pmid for pmid, _ in records}
                job.skipped += len(set(batch) - returned)
                for pmid, data in records:
                    if _cancelled(job):
                        return
                    if job.added >= job.requested:
                        break
                    job.examined += 1
                    temp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as temp:
                            temp.write(data)
                            temp_path = Path(temp.name)
                        try:
                            title, _, meta = parse_document(str(temp_path))
                        except (ET.ParseError, ValueError) as exc:
                            job.skipped += 1
                            print(f'Skipping unusable PMID {pmid}: {exc}', file=sys.stderr, flush=True)
                            continue
                        # Keyword-only records are not scientific abstracts.
                        abstract_body = (meta.get('abstract') or '').split('\n\nKeywords:')[0].strip()
                        if not abstract_body or abstract_body.lower().startswith('keywords:'):
                            job.skipped += 1
                            continue
                        _store_record(job, pmid, data, title, meta, corpus)
                    finally:
                        if temp_path:
                            temp_path.unlink(missing_ok=True)
                _persist(job)
                if job.added >= job.requested:
                    break
            job.offset += len(ids)
            _persist(job)
            if job.offset >= count:
                break
        if job.added >= job.requested:
            _persist(job, status='completed', message=f'Added {job.added:,} unique articles to {job.topic.name}.')
        else:
            _persist(job, status='partial', message=f'Added {job.added:,} of {job.requested:,}. No more eligible new articles were found within the first 10,000 {job.get_source_display()} matches.')
    except ImportCancelled:
        return
    except Exception as exc:
        _persist(job, status='failed', message=f'{exc} Collected articles were kept; retrying will skip existing articles.')
        raise
    finally:
        close_old_connections()
