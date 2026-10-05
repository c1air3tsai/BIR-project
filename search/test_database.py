import tempfile
from pathlib import Path
from unittest.mock import patch

from django.db import OperationalError, connection
from django.test import SimpleTestCase, TestCase, TransactionTestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from .db_retry import retry_sqlite_write
from .models import Document, ImportJob, Posting
from .test_topics import xml
from .topic_import import create_job, run_job, _persist


class LockRetryTests(SimpleTestCase):
    def test_transient_lock_retries_after_returning_to_autocommit(self):
        with patch('search.db_retry.time.sleep'), patch('search.db_retry.close_old_connections'):
            action = __import__('unittest.mock', fromlist=['Mock']).Mock(side_effect=[OperationalError('database is locked'), 'saved'])
            self.assertEqual(retry_sqlite_write(action)(), 'saved')
            self.assertEqual(action.call_count, 2)

    def test_unrelated_error_is_not_retried(self):
        with patch('search.db_retry.time.sleep') as sleep:
            with self.assertRaisesRegex(OperationalError, 'disk I/O error'):
                retry_sqlite_write(lambda: (_ for _ in ()).throw(OperationalError('disk I/O error')))()
            sleep.assert_not_called()

    def test_permanent_lock_has_a_bounded_retry_count(self):
        from unittest.mock import Mock
        action = Mock(side_effect=OperationalError('database is locked'))
        with patch('search.db_retry.time.sleep'), patch('search.db_retry.close_old_connections'):
            with self.assertRaises(OperationalError):
                retry_sqlite_write(action)()
        self.assertEqual(action.call_count, 4)


class ReadOnlyProgressTests(TestCase):
    def test_healthy_progress_poll_does_not_issue_an_update(self):
        job = create_job('GLP-1', 500)
        with CaptureQueriesContext(connection) as queries:
            response = self.client.get(reverse('search:job_status', args=[job.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(any(q['sql'].lstrip().upper().startswith(('UPDATE', 'INSERT', 'DELETE')) for q in queries))

    def test_inner_transaction_does_not_retry_a_broken_statement(self):
        from unittest.mock import Mock
        action = Mock(side_effect=OperationalError('database is locked'))
        with patch('search.db_retry.time.sleep') as sleep:
            with self.assertRaises(OperationalError):
                retry_sqlite_write(action)()
            sleep.assert_not_called()
        self.assertEqual(action.call_count, 1)


class RecordRetryTests(TransactionTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        override = self.settings(BASE_DIR=Path(self.temporary.name))
        override.enable()
        self.addCleanup(override.disable)

    def test_lock_after_indexing_rolls_back_and_retries_the_whole_article(self):
        job = create_job('GLP-1', 1)
        attempts = 0

        def fail_once(current, **fields):
            nonlocal attempts
            if fields.get('message', '').startswith('Adding to'):
                attempts += 1
                if attempts == 1:
                    raise OperationalError('database is locked')
            return _persist(current, **fields)

        with patch('search.topic_import.search_pubmed', return_value=(['901'], 1)), patch('search.topic_import.fetch_pubmed_batch', return_value=[('901', xml(901))]), patch('search.topic_import._persist', side_effect=fail_once), patch('search.db_retry.time.sleep'):
            run_job(job.pk)
        job.refresh_from_db()
        self.assertEqual((job.status, job.imported, job.duplicates), ('completed', 1, 0))
        self.assertEqual(attempts, 2)
        self.assertEqual(Document.objects.count(), 1)
        self.assertEqual(job.topic.documents.count(), 1)
        self.assertTrue(Posting.objects.exists())
        files = list((Path(self.temporary.name) / 'data' / 'corpus').glob('*.xml'))
        self.assertEqual(len(files), 1)

    def test_lock_while_saving_progress_preserves_cancellation_request(self):
        job = create_job('GLP-1', 5)
        ImportJob.objects.filter(pk=job.pk).update(cancel_requested=True)
        real_save = job.save
        with patch.object(job, 'save', side_effect=[OperationalError('database is locked'), None]) as save, patch('search.db_retry.time.sleep'):
            # Use the actual save on the second call.
            count = 0

            def lock_then_save(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 1:
                    raise OperationalError('database is locked')
                return real_save(*args, **kwargs)

            save.side_effect = lock_then_save
            _persist(job, message='Downloading abstracts')
        job.refresh_from_db()
        self.assertTrue(job.cancel_requested)
        self.assertEqual(job.message, 'Downloading abstracts')
