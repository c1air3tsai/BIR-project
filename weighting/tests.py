# =============================================================================
# weighting/tests.py
# Project 3 單元測試：用 4 篇的小語料手算驗證 TF、IDF、TF-IDF、L2 norm、cosine 與評估指標
# 執行：python manage.py test weighting
#
# 手算用的小語料（詞彙順序：glp1, obesity, insulin, semaglutide）
#   d1 = "glp1 obesity obesity"            tf = [1, 2, 0, 0]
#   d2 = "glp1 insulin"                    tf = [1, 0, 1, 0]
#   d3 = "insulin insulin insulin obesity" tf = [0, 1, 3, 0]
#   d4 = "glp1 glp1 semaglutide"           tf = [2, 0, 0, 1]
#   N = 4，df = [3, 2, 2, 1]
#   idf = log10(4/3) = 0.124939, log10(4/2) = 0.301030, 0.301030, log10(4/1) = 0.602060
# =============================================================================
import shutil
import tempfile

import numpy as np
from django.test import TestCase, override_settings
from django.urls import reverse
from scipy import sparse

from search.models import Document

from . import schemes
from .evaluation import metrics_for, ndcg_at, relevance_grades
from .index import load_index
from .preprocessing import PreprocessConfig, preprocess

COUNTS = sparse.csr_matrix(np.array([
    [1, 2, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 3, 0],
    [2, 0, 0, 1],
]))
N = 4


class WeightingMathTests(TestCase):
    def setUp(self):
        self.df = schemes.document_frequency(COUNTS)
        self.idf = schemes.inverse_document_frequency(self.df, N)

    def weights(self, method):
        return schemes.weight_matrix(COUNTS, method, self.idf).toarray()

    def test_document_and_collection_frequency(self):
        # df：每個詞出現在幾篇文件；cf：每個詞總共出現幾次
        self.assertEqual(self.df.tolist(), [3, 2, 2, 1])
        self.assertEqual(schemes.collection_frequency(COUNTS).tolist(), [4, 3, 4, 1])

    def test_idf(self):
        # idf = log10(N / df)
        np.testing.assert_allclose(self.idf, [0.124939, 0.301030, 0.301030, 0.602060], atol=1e-6)
        # 詞不存在（df = 0）時 idf 定義為 0
        self.assertEqual(schemes.inverse_document_frequency([0], N).tolist(), [0.0])

    def test_raw_tf(self):
        np.testing.assert_array_equal(self.weights('raw_tf')[2], [0, 1, 3, 0])

    def test_log_tf(self):
        # d3：insulin tf = 3 → 1 + log10(3) = 1.477121；obesity tf = 1 → 1；tf = 0 → 0
        np.testing.assert_allclose(self.weights('log_tf')[2], [0, 1, 1.477121, 0], atol=1e-6)

    def test_idf_only(self):
        # 只要出現就給 idf，不管出現幾次：d4 的 glp1 出現 2 次，權重仍是 0.124939
        np.testing.assert_allclose(self.weights('idf')[3], [0.124939, 0, 0, 0.602060], atol=1e-6)

    def test_tfidf(self):
        # d1：glp1 = 1 × 0.124939；obesity = 2 × 0.301030 = 0.602060
        np.testing.assert_allclose(self.weights('tfidf')[0], [0.124939, 0.602060, 0, 0], atol=1e-6)

    def test_log_tfidf(self):
        # d3：obesity = 1 × 0.301030；insulin = 1.477121 × 0.301030 = 0.444658
        np.testing.assert_allclose(self.weights('logtfidf')[2], [0, 0.301030, 0.444658, 0], atol=1e-6)

    def test_l2_normalization(self):
        # d1 raw tf = [1, 2]，||d1|| = sqrt(1² + 2²) = sqrt(5) = 2.236068
        normalized = schemes.l2_normalize_rows(schemes.weight_matrix(COUNTS, 'raw_tf', self.idf)).toarray()
        np.testing.assert_allclose(normalized[0], [0.447214, 0.894427, 0, 0], atol=1e-6)
        # 每一列正規化後長度都是 1
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), [1, 1, 1, 1])

    def test_cosine_by_hand(self):
        # q = "obesity insulin" → raw tf [0, 1, 1, 0]；d3 = [0, 1, 3, 0]
        # cos = (1×1 + 1×3) / (sqrt(2) × sqrt(10)) = 4 / 4.472136 = 0.894427
        self.assertAlmostEqual(schemes.cosine([0, 1, 1, 0], [0, 1, 3, 0]), 0.894427, places=6)

    def test_matrix_cosine_matches_formula(self):
        # 矩陣版（L2 後內積）必須等於逐一計算的 cosine
        query = np.array([0, 1, 1, 0], dtype=float)
        for method in schemes.REQUIRED_METHODS:
            doc_matrix = schemes.l2_normalize_rows(schemes.weight_matrix(COUNTS, method, self.idf))
            query_vector = schemes.query_weights(query, method, self.idf)
            scores = schemes.score_documents(doc_matrix, query_vector, use_l2=True)
            dense = schemes.weight_matrix(COUNTS, method, self.idf).toarray()
            expected = [schemes.cosine(query_vector, row) for row in dense]
            np.testing.assert_allclose(scores, expected, atol=1e-12, err_msg=method)

    def test_tfidf_cosine_example(self):
        # q = "obesity"，TF-IDF：q = [0, 0.301030, 0, 0]，d1 = [0.124939, 0.602060, 0, 0]
        # cos = 0.602060 / sqrt(0.124939² + 0.602060²) = 0.602060 / 0.614887 = 0.979139
        query_vector = schemes.query_weights(np.array([0, 1, 0, 0]), 'tfidf', self.idf)
        doc_matrix = schemes.l2_normalize_rows(schemes.weight_matrix(COUNTS, 'tfidf', self.idf))
        self.assertAlmostEqual(schemes.score_documents(doc_matrix, query_vector, True)[0], 0.979139, places=5)

    def test_without_l2_is_dot_product(self):
        # 無 L2：分數 = q · d。q = "insulin"（raw tf [0,0,1,0]）→ d2 = 1、d3 = 3
        scores = schemes.score_documents(schemes.weight_matrix(COUNTS, 'raw_tf', self.idf),
                                         np.array([0, 0, 1, 0], dtype=float), use_l2=False)
        np.testing.assert_allclose(scores, [0, 1, 3, 0])

    def test_rank_scores_order_and_ties(self):
        # 高分在前；同分時列編號小的在前；0 分不列入
        self.assertEqual(schemes.rank_scores(np.array([0.5, 0.0, 0.9, 0.5]), limit=10),
                         [(2, 0.9), (0, 0.5), (3, 0.5)])

    def test_matches_sklearn_for_l2_raw_tf(self):
        # 用 sklearn 驗證（不是主要實作）：TfidfTransformer(use_idf=False, norm='l2') = raw TF + L2
        from sklearn.feature_extraction.text import TfidfTransformer
        expected = TfidfTransformer(use_idf=False, norm='l2').fit_transform(COUNTS).toarray()
        ours = schemes.l2_normalize_rows(schemes.weight_matrix(COUNTS, 'raw_tf', self.idf)).toarray()
        np.testing.assert_allclose(ours, expected)

    def test_bm25_saturates_and_penalises_length(self):
        lengths = np.asarray(COUNTS.sum(axis=1)).ravel()
        bm25 = schemes.weight_matrix(COUNTS, 'bm25', self.idf, lengths, lengths.mean(),
                                     schemes.bm25_idf(self.df, N)).toarray()
        # BM25 權重上限為 idf_BM25 × (k1 + 1)
        upper = schemes.bm25_idf(self.df, N) * (schemes.BM25_K1 + 1)
        self.assertTrue(np.all(bm25 <= upper + 1e-12))


class EvaluationMetricTests(TestCase):
    def test_metrics_by_hand(self):
        # 6 篇文件的相關等級；系統排名 [3, 0, 1, 4, 5, 2]
        grades = np.array([0, 2, 1, 0, 1, 0])
        ranked = [3, 0, 1, 4, 5, 2]
        result = metrics_for(ranked, grades)
        # 前 5 名中相關的有 doc1、doc4 → P@5 = 2/5
        self.assertAlmostEqual(result['p5'], 0.4)
        # 前 10 名（全部 6 篇）相關 3 篇 → P@10 = 3/10，Recall@10 = 3/3
        self.assertAlmostEqual(result['p10'], 0.3)
        self.assertAlmostEqual(result['r10'], 1.0)
        # 第一篇相關文件在第 3 名 → RR = 1/3
        self.assertAlmostEqual(result['mrr'], 1 / 3)

    def test_ndcg_by_hand(self):
        grades = np.array([0, 2, 1, 0, 1, 0])
        # DCG = (2²−1)/log2(4) + (2¹−1)/log2(5) + (2¹−1)/log2(7) = 1.5 + 0.430677 + 0.356207
        # IDCG = 3/log2(2) + 1/log2(3) + 1/log2(4) = 3 + 0.630930 + 0.5
        expected = (1.5 + 0.430677 + 0.356207) / (3 + 0.630930 + 0.5)
        self.assertAlmostEqual(ndcg_at([3, 0, 1, 4, 5, 2], grades, 10), expected, places=5)

    def test_perfect_ranking_has_ndcg_one(self):
        grades = np.array([2, 1, 0])
        self.assertAlmostEqual(ndcg_at([0, 1, 2], grades, 10), 1.0)


class PreprocessingTests(TestCase):
    def test_switches(self):
        text = 'The GLP-1 receptors, and GLP‑1 agonists!'
        # 預設：小寫、去標點、移除 stopword、不 stemming；兩種連字號統一
        self.assertEqual(preprocess(text), ['glp-1', 'receptors', 'glp-1', 'agonists'])
        self.assertEqual(preprocess(text, PreprocessConfig(remove_stopwords=False))[:2], ['the', 'glp-1'])
        self.assertIn('receptor', preprocess(text, PreprocessConfig(stemming=True)))


# 使用暫存資料夾當索引快取，避免動到真正的 data/retrieval_cache
class IndexAndPageTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cache = tempfile.mkdtemp()
        cls.override = override_settings(RETRIEVAL_CACHE_DIR=cls.cache)
        cls.override.enable()

    @classmethod
    def tearDownClass(cls):
        cls.override.disable()
        shutil.rmtree(cls.cache, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        abstracts = ['glp1 obesity obesity', 'glp1 insulin', 'insulin insulin insulin obesity', 'glp1 glp1 semaglutide']
        for number, abstract in enumerate(abstracts, 1):
            Document.objects.create(title=f'Doc {number}', source_file=f'test{number}.xml', pmid=str(9000 + number),
                                    abstract=abstract, raw_text=abstract)

    def test_index_matches_hand_calculation(self):
        index = load_index(force_rebuild=True)
        self.assertEqual(index.n_documents, 4)
        self.assertEqual(index.df[index.term_to_col['glp1']], 3)
        self.assertAlmostEqual(index.idf[index.term_to_col['semaglutide']], 0.602060, places=6)
        # 查詢 "obesity insulin"，Raw TF + L2：d3 cosine = 0.894427 應為第 1 名
        result = index.search('obesity insulin', 'raw_tf', use_l2=True)
        self.assertEqual(index.pmids[result['ranked'][0][0]], '9003')
        self.assertAlmostEqual(result['ranked'][0][1], 0.894427, places=6)

    def test_relevance_rule_uses_mesh(self):
        index = load_index(force_rebuild=True)
        index.mesh = [[{'descriptor': 'Obesity', 'major': True, 'qualifiers': []}],
                      [{'descriptor': 'Insulin', 'major': False, 'qualifiers': ['adverse effects']}], [], []]
        self.assertEqual(relevance_grades(index, {'mesh': ['Obesity']}).tolist(), [2, 0, 0, 0])
        self.assertEqual(relevance_grades(index, {'qualifiers': ['adverse effects']}).tolist(), [0, 1, 0, 0])

    def test_page_and_export(self):
        response = self.client.get(reverse('weighting:page'), {'q': 'insulin', 'method': 'tfidf', 'l2': '0'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<mark>insulin</mark>', html=False)
        export = self.client.get(reverse('weighting:export'))
        self.assertEqual(export.status_code, 200)
        self.assertIn('nDCG@10', export.content.decode('utf-8-sig'))
