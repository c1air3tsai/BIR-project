"""AJAX endpoints for the Term Weighting Lab.  Writes retain Django CSRF checks."""
import json
import logging
import time

from django.http import HttpResponse, JsonResponse
from django.db import IntegrityError
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_GET, require_POST

from .models import Document, EvaluationQuery, RelevanceJudgment
from .retrieval_evaluation import candidate_pool, evaluate
from .term_weighting import METHODS, build_index, index_status, load_index

logger = logging.getLogger(__name__)
SCOPES = ('medical', 'cs', 'all')


def _scope(request):
    value = request.GET.get('corpus', request.POST.get('corpus', 'medical'))
    if value not in SCOPES:
        raise ValueError('Select a valid corpus')
    return value


def _index(request):
    started = time.perf_counter()
    index = load_index(_scope(request))
    logger.debug('term_weighting index_load_seconds=%.5f', time.perf_counter() - started)
    if index is None:
        raise LookupError('Index not ready. Build it first.')
    return index


def _error(exc):
    return JsonResponse({'error': str(exc)}, status=409 if isinstance(exc, LookupError) else 400)


@require_GET
def page(request):
    scope = request.GET.get('corpus', 'medical')
    if scope not in SCOPES:
        scope = 'medical'
    return render(request, 'search/term_weighting.html', {
        'analysis_tab': 'term_weighting', 'page_title': 'Term Weighting · BioMed IR',
        'scope': scope, 'index_status': index_status(scope),
    })


@require_GET
def status(request):
    try:
        return JsonResponse(index_status(_scope(request)))
    except ValueError as exc:
        return _error(exc)


@require_POST
def build(request):
    try:
        return JsonResponse(build_index(_scope(request), force=request.POST.get('force') == '1'))
    except ValueError as exc:
        return _error(exc)


@require_GET
def search(request):
    try:
        index = _index(request)
        query = request.GET.get('q', '').strip()[:300]
        if not query:
            raise ValueError('Enter a query')
        method = request.GET.get('method', 'tfidf')
        result = index.rank(query, method)
        logger.info('term_weighting search scope=%s method=%s timing=%s', index.scope, method, result['timing'])
        return JsonResponse(result)
    except (ValueError, LookupError) as exc:
        return _error(exc)


@require_GET
def explanation(request):
    try:
        index = _index(request)
        return JsonResponse(index.explain(request.GET.get('q', '')[:300],
                                          request.GET.get('method', 'tfidf'),
                                          int(request.GET.get('document_id', '0'))))
    except (ValueError, LookupError) as exc:
        return _error(exc)


@require_GET
def compare(request):
    try:
        index = _index(request)
        query = request.GET.get('q', '').strip()[:300]
        if not query:
            raise ValueError('Enter a query')
        start = time.perf_counter()
        result = index.compare(query)
        result['l2'] = index.l2_experiment(query)
        logger.info('term_weighting comparison scope=%s seconds=%.5f', index.scope, time.perf_counter() - start)
        return JsonResponse(result)
    except (ValueError, LookupError) as exc:
        return _error(exc)


@require_GET
def terms(request):
    try:
        return JsonResponse({'terms': _index(request).inspect_terms(request.GET.get('terms', '')[:300])})
    except (ValueError, LookupError) as exc:
        return _error(exc)


@require_GET
def evaluation(request):
    try:
        start = time.perf_counter()
        result = evaluate(_index(request), include_detail=True)
        logger.info('term_weighting evaluation seconds=%.5f', time.perf_counter() - start)
        return JsonResponse(result)
    except (ValueError, LookupError) as exc:
        return _error(exc)


@require_GET
def pool(request, query_id):
    try:
        index = _index(request)
        query = get_object_or_404(EvaluationQuery, pk=query_id)
        documents, _ = candidate_pool(index, query.text)
        ids = [row['id'] for row in documents]
        judgments = dict(RelevanceJudgment.objects.filter(query=query, document_id__in=ids)
                         .values_list('document_id', 'relevance'))
        return JsonResponse({'query': {'id': query.id, 'text': query.text},
                             'documents': [{**row, 'relevance': judgments.get(row['id'])}
                                           for row in documents], 'source': 'manual'})
    except (ValueError, LookupError) as exc:
        return _error(exc)


@require_POST
def query_change(request):
    action = request.POST.get('action', '')
    try:
        if action == 'add':
            text = request.POST.get('text', '').strip()[:300]
            if not text:
                raise ValueError('Enter query text')
            query = EvaluationQuery.objects.create(text=text)
        else:
            query = get_object_or_404(EvaluationQuery, pk=int(request.POST.get('query_id', '0')))
            if action == 'edit':
                text = request.POST.get('text', '').strip()[:300]
                if not text:
                    raise ValueError('Enter query text')
                if text != query.text:
                    query.text = text
                    query.save(update_fields=['text', 'updated_at'])
                    query.judgments.all().delete()
            elif action == 'delete':
                query.delete()
                return JsonResponse({'deleted': True})
            else:
                raise ValueError('Unknown action')
        return JsonResponse({'id': query.id, 'text': query.text})
    except IntegrityError:
        return _error(ValueError('Query already exists'))
    except (ValueError, OverflowError) as exc:
        return _error(exc)


@require_POST
def judgment(request):
    try:
        index = _index(request)
        query = get_object_or_404(EvaluationQuery, pk=int(request.POST.get('query_id', '0')))
        document_id = int(request.POST.get('document_id', '0'))
        pool_ids = {row['id'] for row in candidate_pool(index, query.text)[0]}
        if document_id not in pool_ids:
            raise ValueError('Document is outside the current judgment pool')
        value = request.POST.get('relevance')
        if value == 'unjudged':
            RelevanceJudgment.objects.filter(query=query, document_id=document_id).delete()
            return JsonResponse({'relevance': None})
        if value not in ('0', '1'):
            raise ValueError('Choose relevant, not relevant, or unjudged')
        document = get_object_or_404(Document, pk=document_id)
        RelevanceJudgment.objects.update_or_create(
            query=query, document=document,
            defaults={'relevance': int(value), 'source': 'manual'})
        return JsonResponse({'relevance': int(value)})
    except (ValueError, LookupError) as exc:
        return _error(exc)


@require_GET
def export(request):
    try:
        index = _index(request)
        result = evaluate(index, include_detail=True)
        payload = {'config': {'corpus': index.scope, 'preprocessing': 'Condition C; hyphen variants folded',
                              'methods': METHODS, 'idf': 'ln(N/DF)', 'score': 'cosine', 'index_key': index.key},
                   'documents': index.n, 'vocabulary': len(index.df), **result}
        response = HttpResponse(json.dumps(payload, ensure_ascii=False, indent=2), content_type='application/json')
        response['Content-Disposition'] = 'attachment; filename="term_weighting_results.json"'
        return response
    except (ValueError, LookupError) as exc:
        return _error(exc)
