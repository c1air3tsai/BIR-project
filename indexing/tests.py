# =============================================================================
# indexing/tests.py
# Project 5 單元測試：python manage.py test indexing
#
# 小語料就是作業 Part 1 的範例表格：
#   information D1, D3, D5 / retrieval D1, D2, D5 / system D2, D3, D4 / algorithm D1, D4
# 記憶體上限故意設得很小，讓 BSBI、SPIMI 都切成好幾個 blocks，檢查合併後和基準索引完全相同
# =============================================================================
import tempfile
from pathlib import Path
from unittest import skipUnless

from django.test import SimpleTestCase
from django.urls import reverse

from reuters.corpus import cache_path

from . import baseline, bsbi, sources, spimi, views
from .collection import corpus_size, read_documents
from .inverted_index import DiskIndex, InvertedIndex, intersect, union
from .storage import read_term_block, read_termid_block

DOCS = [(1, 'Information retrieval algorithm'), (2, 'retrieval system retrieval'), (3, 'information system'),
        (4, 'system algorithm'), (5, 'information retrieval')]
TABLE = {'information': [1, 3, 5], 'retrieval': [1, 2, 5], 'system': [2, 3, 4], 'algorithm': [1, 4]}


class BooleanTests(SimpleTestCase):
    def test_intersect_and_union(self):
        self.assertEqual(intersect([1, 3, 5, 9], [2, 3, 9, 10]), [3, 9])
        self.assertEqual(union([1, 3, 5], [2, 3, 6]), [1, 2, 3, 5, 6])
        self.assertEqual(intersect([], [1, 2]), [])
        self.assertEqual(union([], [1, 2]), [1, 2])


class HighlightTests(SimpleTestCase):
    def test_marks_whole_tokens_only(self):
        # 和索引的比對規則一致：Oil、OIL 會標；oil-related、oils 在索引裡是不同的 term，不標；< 要 escape
        counts = {'oil': 0, 'price': 0}
        html = views._mark('Oil prices <up>: OIL price, oil-related oils', counts)
        self.assertEqual(html, '<mark>Oil</mark> prices &lt;up&gt;: <mark>OIL</mark> <mark>price</mark>, oil-related oils')
        self.assertEqual(counts, {'oil': 2, 'price': 1})


class InvertedIndexTests(SimpleTestCase):
    def test_assignment_table(self):
        index = InvertedIndex(DOCS)
        self.assertEqual(index.vocabulary(), sorted(TABLE))
        for term, postings in TABLE.items():
            self.assertEqual(index.postings(term), postings)
        self.assertEqual(index.and_query('information', 'retrieval'), [1, 5])
        self.assertEqual(index.or_query('system', 'algorithm'), [1, 2, 3, 4])
        self.assertEqual(index.postings('missing'), [])


class ConstructionTests(SimpleTestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())

    def _check(self, result, folder):
        disk = DiskIndex(folder / 'index')
        self.assertEqual(disk.vocabulary(), sorted(TABLE))
        for term, postings in TABLE.items():
            self.assertEqual(list(disk.postings(term)), postings)
        self.assertEqual(disk.and_query('information', 'retrieval'), [1, 5])
        self.assertGreater(len(result['blocks']), 1)

    def test_bsbi_blocks_and_merge(self):
        # 每個 block 最多約 3 個 pairs → 一定會切成好幾個 blocks
        result = bsbi.build(DOCS, self.folder / 'bsbi', 3 * bsbi.PAIR_BYTES, samples=True, trace_terms=('retrieval',))
        self._check(result, self.folder / 'bsbi')
        # 不同 blocks 的 termID 一致：每個 block 檔裡 retrieval 都是同一個 termID
        term_id = result['samples']['term_ids']['retrieval']
        found = [tid for path in sorted((self.folder / 'bsbi' / 'blocks').glob('*.bin'))
                 for tid, _ in read_termid_block(path) if tid == term_id]
        self.assertEqual(len(found), len(result['samples']['trace']['retrieval']))
        # pairs 有排序、同一篇的重複 pair 在局部索引裡只剩一個（D2 有兩次 retrieval）
        pairs = [(tid, doc) for tid, _, doc in result['samples']['sorted']]
        self.assertEqual(pairs, sorted(pairs))

    def test_baseline_writes_same_index(self):
        result = baseline.build(DOCS, self.folder / 'baseline')
        disk = DiskIndex(self.folder / 'baseline' / 'index')
        self.assertEqual(disk.vocabulary(), sorted(TABLE))
        for term, postings in TABLE.items():
            self.assertEqual(list(disk.postings(term)), postings)
        self.assertEqual(result['blocks'], [])
        self.assertGreater(result['index_memory'], 0)

    def test_spimi_blocks_and_merge(self):
        result = spimi.build(DOCS, self.folder / 'spimi', 300)
        self._check(result, self.folder / 'spimi')
        for path in (self.folder / 'spimi' / 'blocks').glob('*.bin'):
            terms = [term for term, _ in read_term_block(path)]
            self.assertEqual(terms, sorted(terms))
            for _, postings in read_term_block(path):
                self.assertEqual(list(postings), sorted(set(postings)))


@skipUnless(cache_path().is_file(), 'Reuters-21578 尚未下載（先跑 run_reuters_experiments）')
class ReutersTests(SimpleTestCase):
    def test_bsbi_equals_spimi_on_reuters(self):
        folder = Path(tempfile.mkdtemp())
        baseline = InvertedIndex(read_documents(2000))
        for module in (bsbi, spimi):
            module.build(read_documents(2000), folder / module.__name__, 256 * 1024)
            disk = DiskIndex(folder / module.__name__ / 'index')
            self.assertEqual(disk.vocabulary(), baseline.vocabulary())
            self.assertTrue(all(list(disk.postings(t)) == baseline.postings(t) for t in baseline.index))

    def test_document_popup_highlights_query_terms(self):
        response = self.client.get(reverse('indexing:document', args=[2]), {'terms': 'Oil price'})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        counts = {row['term']: row['count'] for row in data['counts']}
        html = data['title'] + ''.join(data['paragraphs'])
        self.assertGreater(counts['oil'], 0)                     # oil 的 postings 第一篇就是 Doc 2
        self.assertEqual(html.lower().count('<mark>oil</mark>'), counts['oil'])
        self.assertEqual(self.client.get(reverse('indexing:document', args=[999999])).status_code, 404)

    @skipUnless(sources.PUBMED_FILE.is_file(), 'PubMed GLP-1 尚未下載（先跑 fetch_pubmed_glp1）')
    def test_pubmed_source(self):
        # 第二個資料來源：docID 是 1–N 連續編號，BSBI 與 SPIMI 在小上限下建出和基準索引相同的索引
        ids = [doc_id for doc_id, _ in read_documents(source='pubmed')]
        self.assertEqual(ids, list(range(1, corpus_size('pubmed') + 1)))
        folder = Path(tempfile.mkdtemp())
        baseline = InvertedIndex(read_documents(1000, 'pubmed'))
        for module in (bsbi, spimi):
            result = module.build(read_documents(1000, 'pubmed'), folder / module.__name__, 256 * 1024)
            self.assertGreater(len(result['blocks']), 1)
            disk = DiskIndex(folder / module.__name__ / 'index')
            self.assertTrue(all(list(disk.postings(t)) == baseline.postings(t) for t in baseline.index))
        response = self.client.get(reverse('indexing:document', args=[1]), {'source': 'pubmed', 'terms': 'glp-1'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['meta'][0].startswith('PMID'))

    def test_repeated_corpus_gets_new_doc_ids(self):
        size = corpus_size()
        ids = [doc_id for doc_id, _ in read_documents(size + 3)]
        self.assertEqual(len(ids), len(set(ids)))
        # 第二輪的 docID = 原本的 docID + 最後一篇的 NEWID
        self.assertEqual(ids[size:], [doc_id + ids[size - 1] for doc_id in ids[:3]])
