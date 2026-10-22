import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.test import Client, SimpleTestCase, TestCase
from django.urls import reverse

from .models import Document, EvaluationQuery, RelevanceJudgment
from .retrieval_evaluation import dcg_at, evaluate, metrics, ndcg_at
from .term_weighting import (
    METHODS, _load_artifact, build_index, idf_weight, index_status, load_index, log_tf,
    raw_tf, tokens, weight,
)


class FormulaTests(SimpleTestCase):
    def test_weighting_by_hand(self):
        value = math.log(3 / 2)
        self.assertEqual(raw_tf(3), 3)
        self.assertEqual(log_tf(0), 0)
        self.assertAlmostEqual(log_tf(3), 1 + math.log(3))
        self.assertAlmostEqual(idf_weight(3, 2), value)
        self.assertEqual(idf_weight(3, 3), 0)
        self.assertAlmostEqual(weight(3, value, 'tfidf'), 3 * value)
        self.assertAlmostEqual(weight(3, value, 'idf'), value)
        self.assertEqual(weight(0, value, 'idf'), 0)

    def test_condition_c_and_scientific_connectors(self):
        self.assertEqual(tokens('The GLP-1 GLP‐1 GLP–1 semaglutide'),
                         ['glp-1', 'glp-1', 'glp-1', 'semaglutide'])

    def test_metrics_by_hand_and_edge_cases(self):
        qrels = {1: 1, 2: 0, 3: 1, 4: 0, 5: 1}
        result = metrics([2, 4, 1, 3, 5], qrels)
        self.assertAlmostEqual(result['p5'], 3 / 5)
        self.assertAlmostEqual(result['p10'], 3 / 10)
        self.assertAlmostEqual(result['r10'], 1)
        self.assertAlmostEqual(result['mrr'], 1 / 3)
        expected_dcg = 1 / math.log2(4) + 1 / math.log2(5) + 1 / math.log2(6)
        self.assertAlmostEqual(dcg_at([2, 4, 1, 3, 5], qrels, 10), expected_dcg)
        ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(result['ndcg10'], expected_dcg / ideal)
        self.assertEqual(metrics([2, 4], qrels)['p10'], 0)
        self.assertEqual(metrics([2, 4], qrels)['mrr'], 0)
        self.assertEqual(metrics([1], qrels)['mrr'], 1)
        self.assertEqual(metrics([2, 4, 1, 3, 5], {})['r10'], 0)
        self.assertEqual(ndcg_at([2, 4], {}, 10), 0)


class RetrievalTests(TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        override = self.settings(RETRIEVAL_CACHE_DIR=Path(directory.name))
        override.enable()
        self.addCleanup(override.disable)
        self.docs = []
        for i, abstract in enumerate(('alpha alpha beta', 'alpha gamma', 'beta gamma'), 1):
            self.docs.append(Document.objects.create(
                title=f'Paper {i}', source_file=f'tw-{i}.xml', abstract=abstract,
                raw_text=abstract, publication_year='2024'))
        EvaluationQuery.objects.all().delete()

    def test_index_cosine_explanation_and_cache(self):
        built = build_index()
        self.assertFalse(built['cache_hit'])
        self.assertEqual(built['indexed'], 3)
        self.assertEqual(built['vocabulary'], 3)
        self.assertTrue(build_index()['cache_hit'])
        index = load_index()
        self.assertEqual(index.df['alpha'], 2)
        self.assertEqual(index.cf['alpha'], 3)
        self.assertAlmostEqual(index.idf['alpha'], math.log(3 / 2))
        result = index.rank('alpha beta', 'tf')
        self.assertEqual(result['results'][0]['id'], self.docs[0].id)
        self.assertEqual([row['score'] for row in result['results']],
                         sorted((row['score'] for row in result['results']), reverse=True))
        self.assertAlmostEqual(result['results'][0]['score'], 3 / math.sqrt(10))
        detail = index.explain('alpha beta', 'tf', self.docs[0].id)
        self.assertEqual(detail['terms'][0]['tf'], 2)
        self.assertEqual(detail['terms'][0]['contribution'], 2)
        self.assertAlmostEqual(detail['score'], result['results'][0]['score'])
        self.assertAlmostEqual(index.norms['tf'][0], math.sqrt(5))
        self.assertAlmostEqual(index.norms['log_tf'][0], math.sqrt((1 + math.log(2)) ** 2 + 1))
        self.assertNotEqual(index.norms['tf'][0], index.norms['log_tf'][0])
        self.assertEqual(index.rank('unseenalpha')['results'], [])
        self.assertEqual(index.rank('unseenalpha')['oov'], ['unseenalpha'])
        self.assertEqual(index.rank('the and')['results'], [])
        l2 = index.l2_experiment('alpha beta')
        self.assertTrue(l2['same_order'])
        self.assertEqual(l2['overlap'], len(index.rank('alpha beta')['results']))
        self.assertLess(l2['max_score_difference'], 1e-12)
        self.assertEqual(index.compare('alpha beta')['methods'], list(METHODS))
        self.assertLessEqual(len(index.rank('alpha', limit=10)['results']), 10)
        index_key = index.key
        _load_artifact.cache_clear()
        self.assertEqual(load_index().rank('alpha beta', 'tf')['results'], result['results'])
        self.docs[0].abstract = 'delta'
        self.docs[0].save()
        self.assertFalse(index_status('medical')['ready'])
        self.assertNotEqual(build_index()['key'], index_key)

    def test_scope_and_api_without_search_rebuilding(self):
        self.docs[2].arxiv_id = '2401.00001'
        self.docs[2].save()
        self.assertEqual(build_index('medical')['indexed'], 2)
        self.assertEqual(build_index('cs')['indexed'], 1)
        self.assertEqual(build_index('all')['indexed'], 3)
        page = self.client.get(reverse('search:term_weighting'))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, 'Term Weighting Lab')
        self.assertContains(page, reverse('search:distance'))
        with patch('search.term_weighting_views.build_index', side_effect=AssertionError('unexpected rebuild')):
            result = self.client.get(reverse('search:term_weighting_search'), {'q': 'alpha', 'method': 'tf'})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['results'][0]['id'], self.docs[0].id)
        for method in METHODS:
            self.assertEqual(self.client.get(reverse('search:term_weighting_search'),
                                             {'q': 'alpha', 'method': method}).status_code, 200)
        comparison = self.client.get(reverse('search:term_weighting_compare'), {'q': 'alpha beta'}).json()
        self.assertEqual(comparison['term_evidence']['document_id'], self.docs[0].id)
        self.assertEqual(comparison['l2']['overlap'], len(comparison['rankings']['tfidf']))

    def test_query_document_preprocessing_and_csrf(self):
        self.docs[0].abstract = 'GLP‐1 GLP–1 semaglutide'
        self.docs[0].save()
        build_index()
        index = load_index()
        self.assertEqual(index.cf['glp-1'], 2)
        self.assertEqual(index.rank('GLP-1')['results'][0]['id'], self.docs[0].id)
        client = Client(enforce_csrf_checks=True)
        self.assertEqual(client.get(reverse('search:term_weighting')).status_code, 200)
        token = client.cookies['csrftoken'].value
        response = client.post(reverse('search:term_weighting_queries'),
                               {'action': 'add', 'text': 'semaglutide'},
                               HTTP_X_CSRFTOKEN=token)
        self.assertEqual(response.status_code, 200)

    def test_judgments_are_manual_persistent_and_incomplete_excluded(self):
        build_index()
        query = EvaluationQuery.objects.create(text='alpha beta')
        other = EvaluationQuery.objects.create(text='gamma')
        index = load_index()
        pool = index.compare(query.text)['rows']
        self.assertGreater(len(pool), 0)
        self.assertEqual(evaluate(index)['ready'], 0)
        for row in pool:
            response = self.client.post(reverse('search:term_weighting_judgment'),
                                        {'query_id': query.id, 'document_id': row['id'],
                                         'relevance': '1' if row['id'] == self.docs[0].id else '0'})
            self.assertEqual(response.status_code, 200)
        result = evaluate(index)
        self.assertEqual(result['ready'], 1)
        self.assertEqual(result['total'], 2)
        self.assertEqual(result['aggregate']['tf']['p10'], result['queries'][0]['metrics']['tf']['p10'])
        self.assertEqual(RelevanceJudgment.objects.get(query=query, document=self.docs[0]).source, 'manual')
        self.assertEqual(self.client.get(reverse('search:term_weighting_evaluation')).json()['ready'], 1)
        exported = json.loads(self.client.get(reverse('search:term_weighting_export')).content)
        self.assertEqual(exported['aggregate'], self.client.get(reverse('search:term_weighting_evaluation')).json()['aggregate'])
        self.assertEqual(exported['queries'][0]['judged'], len(pool))
        self.assertEqual(RelevanceJudgment.objects.filter(query=other).count(), 0)
        response = self.client.post(reverse('search:term_weighting_queries'),
                                    {'action': 'edit', 'query_id': query.id, 'text': 'new alpha query'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(RelevanceJudgment.objects.filter(query=query).count(), 0)

    def test_ten_query_mean(self):
        build_index()
        for i in range(10):
            query = EvaluationQuery.objects.create(text=f'alpha {i}')
            for row in load_index().compare(query.text)['rows']:
                RelevanceJudgment.objects.create(query=query, document_id=row['id'],
                                                 relevance=int(row['id'] == self.docs[0].id), source='manual')
        result = evaluate(load_index())
        self.assertEqual(result['ready'], 10)
        for method in METHODS:
            self.assertAlmostEqual(result['aggregate'][method]['p5'],
                                   sum(row['metrics'][method]['p5'] for row in result['queries']) / 10)
