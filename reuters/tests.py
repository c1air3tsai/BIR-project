# =============================================================================
# reuters/tests.py
# Project 4 單元測試：python manage.py test reuters
#   1. SGML 解析與 ModApte 切分規則
#   2. 前處理（去數字、stopword、Reuter 署名）
#   3. TF-IDF 加權平均詞向量的公式（手算）
#   4. 自製 numpy CBOW 的梯度（與數值微分比對）與 loss 會下降
#   5. 評估：topic 相關性
#   6. 網頁（若 Reuters 資料與模型已準備好）
# =============================================================================
import tempfile
from pathlib import Path
from unittest import skipUnless

import numpy as np
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from .corpus import cache_path, modapte_split, parse_sgml
from .retrieval import ReutersIndex
from .text import preprocess, sentences
from .toy_word2vec import ToyWord2Vec, build_vocabulary, softmax, training_pairs

SGML = '''<!DOCTYPE lewis SYSTEM "lewis.dtd">
<REUTERS TOPICS="YES" LEWISSPLIT="TRAIN" CGISPLIT="TRAINING-SET" OLDID="1" NEWID="1">
<DATE>26-FEB-1987 15:01:01.79</DATE><TOPICS><D>crude</D><D>nat-gas</D></TOPICS>
<TEXT><TITLE>OPEC RAISES OUTPUT</TITLE><BODY>OPEC said it raised crude &lt;oil&gt; output by 12 pct.
 Reuter
&#3;</BODY></TEXT></REUTERS>
<REUTERS TOPICS="YES" LEWISSPLIT="TEST" CGISPLIT="TRAINING-SET" OLDID="2" NEWID="2">
<DATE>9-APR-1987</DATE><TOPICS></TOPICS><TEXT><TITLE>NO TOPIC STORY</TITLE><BODY>Text.</BODY></TEXT></REUTERS>
<REUTERS TOPICS="NO" LEWISSPLIT="TRAIN" CGISPLIT="TRAINING-SET" OLDID="3" NEWID="3">
<DATE>1-MAR-1987</DATE><TOPICS><D>earn</D></TOPICS><TEXT><TITLE>X</TITLE></TEXT></REUTERS>
'''


class CorpusTests(SimpleTestCase):
    def test_parse_and_modapte_split(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'reut2-000.sgm').write_text(SGML, encoding='latin-1')
            docs = parse_sgml(directory)
        self.assertEqual([d['id'] for d in docs], [1, 2, 3])
        self.assertEqual(docs[0]['topics'], ['crude', 'nat-gas'])
        # &lt; 還原、結尾的 Reuter 署名與控制字元被移除
        self.assertEqual(docs[0]['body'], 'OPEC said it raised crude <oil> output by 12 pct.')
        # README：TEST + TOPICS=YES 即使沒有任何 topic 仍屬於測試集；TOPICS=NO 不使用
        self.assertEqual([d['split'] for d in docs], ['train', 'test', 'unused'])

    def test_split_rule(self):
        self.assertEqual(modapte_split({'TOPICS': 'YES', 'LEWISSPLIT': 'TRAIN'}), 'train')
        self.assertEqual(modapte_split({'TOPICS': 'BYPASS', 'LEWISSPLIT': 'TRAIN'}), 'unused')
        self.assertEqual(modapte_split({'TOPICS': 'YES', 'LEWISSPLIT': 'NOT-USED'}), 'unused')


class TextTests(SimpleTestCase):
    def test_preprocess_removes_numbers_and_stopwords(self):
        self.assertEqual(preprocess('The OPEC output rose 12 pct to 1.5 mln barrels'),
                         ['opec', 'output', 'rose', 'pct', 'mln', 'barrels'])

    def test_sentences_ignore_line_breaks(self):
        # Reuters 內文在句子中間換行，不能被當成句子邊界
        self.assertEqual(len(sentences('Wheat exports rose\nsharply in March. Corn fell.')), 2)


# 用假的 3 維詞向量手算 TF-IDF 加權平均
class WeightedAverageTests(SimpleTestCase):
    def test_tfidf_weighted_average_by_hand(self):
        table = {'oil': np.array([1.0, 0.0, 0.0]), 'crude': np.array([0.0, 1.0, 0.0])}

        class Vectors:
            vector_size = 3
            key_to_index = {'oil': 0, 'crude': 1}

            def __getitem__(self, word):
                return table[word]

        # 權重 oil = 3、crude = 1、unknown 沒有向量被忽略 → (3·[1,0,0] + 1·[0,1,0]) / 4
        result = ReutersIndex._average({'oil': 3.0, 'crude': 1.0, 'unknown': 5.0}, Vectors())
        np.testing.assert_allclose(result, [0.75, 0.25, 0.0])

    def test_empty_document_gives_zero_vector(self):
        class Vectors:
            vector_size = 2
            key_to_index = {}
        np.testing.assert_array_equal(ReutersIndex._average({'x': 1.0}, Vectors()), [0.0, 0.0])


class ToyWord2VecTests(SimpleTestCase):
    def setUp(self):
        self.sentences = [s.split() for s in ['opec raised oil output', 'opec cut oil prices', 'wheat exports rose']]
        self.vocabulary = build_vocabulary(self.sentences)
        self.pairs = training_pairs(self.sentences, self.vocabulary, window=1)

    def test_training_pairs(self):
        first = self.pairs[0]
        v = self.vocabulary
        # 'opec' 是句首，window=1 時上下文只有右邊的 'raised'
        self.assertEqual(first, ([v['raised']], v['opec']))

    def test_cbow_gradient_matches_numerical_gradient(self):
        model = ToyWord2Vec(self.vocabulary, dim=4, seed=1)
        context, center = self.pairs[2]

        def loss(W_out):
            h = model.W_in[context].mean(axis=0)
            return -np.log(softmax(h @ W_out)[center])

        h = model.W_in[context].mean(axis=0)
        p = softmax(h @ model.W_out)
        e = p.copy()
        e[center] -= 1.0
        analytic = np.outer(h, e)                       # ∂loss/∂W_out = h ⊗ (p − onehot)
        numeric = np.zeros_like(model.W_out)
        eps = 1e-6
        for i in range(model.W_out.shape[0]):
            for j in range(model.W_out.shape[1]):
                plus, minus = model.W_out.copy(), model.W_out.copy()
                plus[i, j] += eps
                minus[i, j] -= eps
                numeric[i, j] = (loss(plus) - loss(minus)) / (2 * eps)
        np.testing.assert_allclose(analytic, numeric, atol=1e-6)

    def test_loss_decreases_for_both_architectures(self):
        for architecture in ('cbow', 'skipgram'):
            losses = ToyWord2Vec(self.vocabulary, dim=5).train(self.pairs, architecture, epochs=60, lr=0.1)
            self.assertLess(losses[-1], losses[0] * 0.7, architecture)


# 需要真實資料與模型：Docker 或本機執行過 run_reuters_experiments 之後才會跑
@skipUnless(cache_path().is_file(), 'Reuters corpus not prepared')
class ReutersDataTests(TestCase):
    def test_modapte_counts_match_readme(self):
        from .corpus import load_documents
        counts = {}
        for doc in load_documents():
            counts[doc['split']] = counts.get(doc['split'], 0) + 1
        self.assertEqual(counts, {'train': 9603, 'test': 3299, 'unused': 8676})

    def test_relevance_and_page(self):
        from .evaluation import relevance
        from .retrieval import load_index
        index = load_index('train')
        self.assertEqual(int(relevance(index, 'crude').sum()), 189)
        response = self.client.get(reverse('reuters:page'), {'q': 'wheat exports', 'method': 'tfidf', 'topic': 'grain'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<mark>wheat</mark>', html=False)

    def test_news_detail_shows_ranks_and_highlight(self):
        from .retrieval import load_index
        index = load_index('train')
        top_row, _ = index.ranking('wheat exports', 'tfidf', limit=1)[0]
        newid = index.documents[top_row]['id']
        response = self.client.get(reverse('reuters:news', args=[newid]),
                                   {'q': 'wheat exports', 'method': 'tfidf', 'topic': 'grain'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '#1')                      # 方法 A 的第 1 名
        self.assertContains(response, '<mark>', html=False)
        self.assertEqual(self.client.get(reverse('reuters:news', args=[999999])).status_code, 404)
