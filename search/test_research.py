import io
import json
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch
from django.test import TestCase, SimpleTestCase
from django.urls import reverse
from django.http import QueryDict
from django.core.files.uploadedfile import SimpleUploadedFile
from .csv_import import import_abstract_csv
from .models import Document
from .forms import get_or_create_topic
from .analysis import analyze
from .domain_analysis import compare_domains
from . import embeddings
from .distance import normalize, edit_distance
from .research_report import report
from .arxiv_client import search_arxiv
from .indexer import index_file
from .topic_import import create_job, run_job

ATOM=b'''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:open="http://a9.com/-/spec/opensearch/1.1/"><open:totalResults>1</open:totalResults><entry><id>http://arxiv.org/abs/2401.12345v2</id><title>Learning models</title><summary>Learning models improve retrieval. Algorithms rank documents.</summary><published>2024-01-05T00:00:00Z</published><author><name>Ada Test</name></author><category term="cs.IR"/></entry></feed>'''

class UnicodeTests(SimpleTestCase):
    def test_levenshtein_operations_and_unicode(self):
        self.assertEqual(edit_distance('kitten','sitting'),3)
        self.assertEqual(edit_distance('','abc'),3)
        self.assertEqual(edit_distance('ab','ba'),2)
        self.assertNotEqual(edit_distance('é','e\u0301'),0)
        self.assertEqual(edit_distance(normalize('é'),normalize('e\u0301')),0)
        self.assertEqual(normalize('Straße'),normalize('STRASSE'))
        self.assertNotEqual(normalize('Ａ','NFC'),normalize('A','NFC'))
        self.assertEqual(normalize('Ａ','NFKC'),normalize('A','NFKC'))

class ResearchFeaturesTests(TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.directory=Path(self.temp.name)
        override=self.settings(BASE_DIR=self.directory);override.enable();self.addCleanup(override.disable);self.addCleanup(self.temp.cleanup)
        self.med=get_or_create_topic('Medical');self.cs=get_or_create_topic('CS')
        self.docs=[]
        for i in range(6):
            text='Patients receive treatment. Treatment helps patients and improves outcomes.' if i<3 else 'Learning models rank documents. Retrieval algorithms improve learning models.'
            d=Document.objects.create(title=f'Article {i}',source_file=f'{i}.xml',abstract=text,raw_text=text,pmid=str(100+i) if i<3 else '',arxiv_id=f'2401.0000{i}' if i>=3 else '')
            d.topics.add(self.med if i<3 else self.cs);self.docs.append(d)

    def params(self,**changes):
        return {**embeddings.DEFAULTS,'epochs':10,'dimensions':50,'min_count':1,**changes}

    def test_domain_balancing_overlap_and_dedup(self):
        self.docs[0].topics.add(self.cs)
        q=QueryDict('',mutable=True);q.setlist('domain_a',[str(self.med.pk)]);q.setlist('domain_b',[str(self.cs.pk)]);q['sample']='500'
        result=compare_domains(q)
        self.assertEqual(result['overlap_count'],1)
        self.assertEqual([g['data']['documents'] for g in result['domain_groups']],[2,2])
        self.assertFalse(set(result['domain_groups'][0]['document_ids'])&set(result['domain_groups'][1]['document_ids']))
        q['sample']='all';q['overlap']='include'
        result=compare_domains(q);self.assertEqual([g['data']['documents'] for g in result['domain_groups']],[3,4])

    def test_actual_training_cosine_repeatability_and_corpus_invalidation(self):
        import numpy as np
        records=embeddings.corpus_records(Document.objects.all());params=self.params()
        first=embeddings.train(records,params);original=first.wv['learning'].copy()
        word,rows=embeddings.neighbors(first,'LEARNING','C')
        self.assertEqual(word,'learning');self.assertTrue(rows)
        candidate=rows[0]['term'];expected=float(np.dot(first.wv[word],first.wv[candidate])/(np.linalg.norm(first.wv[word])*np.linalg.norm(first.wv[candidate])))
        self.assertAlmostEqual(rows[0]['similarity'],expected,places=6)
        second=embeddings.train(records,params);np.testing.assert_allclose(original,second.wv['learning'],atol=0)
        self.assertIsNotNone(embeddings.load(records,params))
        self.assertIsNone(embeddings.load(records,self.params(condition='B')))
        self.assertIsNone(embeddings.load(records+[(999,'New abstract.')],params))
        self.assertEqual(len(embeddings.projection(second,[word,candidate])['points']),2)
        overview=embeddings.vocabulary_projection(second,count=5)
        self.assertEqual(len(overview['points']),5)
        self.assertEqual(overview['points'][0]['rank'],1)
        self.assertGreater(overview['points'][0]['count'],0)
        with self.assertRaisesRegex(ValueError,'outside'):embeddings.neighbors(first,'unknownzz','C')

    def test_pages_reports_exports_and_train_post(self):
        for name in ['domains','word2vec','research','distance']:
            response=self.client.get(reverse('search:'+name));self.assertEqual(response.status_code,200,name)
        zipf=self.client.get(reverse('search:zipf'),{'condition':'all'})
        self.assertContains(zipf,'data-chart-svg="frequency"')
        self.assertContains(zipf,'data-chart-svg="log"')
        self.assertEqual(zipf.content.count(b'data-series checked'),4)
        self.assertNotContains(zipf,'data-chart-mode="residual"')
        domains=self.client.get(reverse('search:domains'),{'domain_a':self.med.pk,'domain_b':self.cs.pk})
        self.assertContains(domains,'data-chart-svg="frequency"')
        self.assertContains(domains,'data-chart-svg="log"')
        self.assertEqual(domains.content.count(b'data-series checked'),2)
        self.assertContains(domains,'search/zipf_dual.js')
        home=self.client.get(reverse('search:home'))
        self.assertContains(home,'data-set-language="en"')
        self.assertContains(home,'data-set-language="zh-Hant"')
        self.assertContains(home,'search/i18n.js')
        response=self.client.get(reverse('search:research'),{'topic':self.med.pk,'report_term':'treatment','condition':'C'})
        self.assertEqual(response.context['selected']['documents'],3)
        self.assertEqual(response.context['tfidf_page'][0]['tf'],2)
        self.assertEqual(response.context['tfidf_page'][0]['idf'],0)
        words=response.context['discussion_words'];self.assertTrue(300<=words<=500,words)
        response=self.client.post(reverse('search:word2vec')+'?topic='+str(self.cs.pk),{**self.params(),'topic':self.cs.pk,'word':'learning'})
        self.assertEqual(response.status_code,302)
        response=self.client.get(response.url);self.assertEqual(response.context['model_metadata']['documents'],3);self.assertTrue(response.context['neighbors']);self.assertTrue(response.context['overview_projection']['points'])
        self.assertContains(response,'id="vocabulary-pca-chart"')
        self.assertContains(response,'id="vocabulary-pca-data"')
        exported=self.client.get(reverse('search:word2vec_export'),{**self.params(),'topic':self.cs.pk,'word':'learning'})
        with zipfile.ZipFile(io.BytesIO(exported.content)) as z:
            self.assertIn('similar_words.csv',z.namelist());self.assertIn('vocabulary_pca.json',z.namelist());self.assertTrue(z.read('vectors.txt').startswith(b'8 50') or z.read('vectors.txt').splitlines()[0].endswith(b' 50'))
        for name,required in [('domains_export','domains_summary.csv'),('research_export','tfidf.csv')]:
            response=self.client.get(reverse('search:'+name));self.assertEqual(response.status_code,200)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:self.assertIn(required,z.namelist())

    def test_no_get_training_and_insufficient_vocabulary(self):
        with patch('search.embeddings.train') as train:
            self.client.get(reverse('search:word2vec'));train.assert_not_called()
        with self.assertRaises(ValueError):embeddings.train([(1,'X.')],self.params(min_count=5))

    def test_arxiv_metadata_import_has_real_id_no_invented_pmid_and_reuses_versions(self):
        with patch('search.arxiv_client._get',return_value=ATOM) as fetch, patch('search.arxiv_client.time.sleep'):
            records,count=search_arxiv('cat:cs.IR');self.assertEqual(count,1)
            self.assertEqual(records[0][0],'2401.12345');self.assertEqual(fetch.call_args.kwargs['min_interval'],3.1)
        path=self.directory/'article.xml';path.write_bytes(records[0][1]);doc=index_file(str(path))
        self.assertEqual(doc.arxiv_id,'2401.12345');self.assertEqual(doc.pmid,'');self.assertEqual(doc.authors,'Ada Test')
        job=create_job('cat:cs.IR',1,source='arxiv')
        with patch('search.topic_import.search_arxiv',return_value=(records,1)):run_job(job.pk)
        job.refresh_from_db();self.assertEqual(job.status,'completed');self.assertEqual(job.linked,1)
        self.assertEqual(Document.objects.filter(arxiv_id='2401.12345').count(),1)

    def test_csv_arxiv_fallback_reuses_versions_and_reports_invalid_rows(self):
        uploaded=SimpleUploadedFile('cs.csv',b'id,title,abstract\n2401.12345v1,CS article,Learning models rank documents.\n2401.12345v2,CS article,Learning models rank documents.\nbad,Invalid,No valid identifier.\n')
        result=import_abstract_csv(uploaded,self.cs)
        self.assertEqual(result['new'],1);self.assertEqual(result['reused'],1);self.assertEqual(len(result['errors']),1)
        doc=Document.objects.get(arxiv_id='2401.12345');self.assertEqual(doc.pmid,'');self.assertEqual(doc.topics.get(),self.cs)
        self.assertEqual(len(list((self.directory/'data'/'corpus').glob('*.xml'))),1)

    def test_clearing_both_domains_does_not_restore_defaults(self):
        q=QueryDict('condition=D&sample=500&overlap=exclude')
        result=compare_domains(q)
        self.assertFalse(result['comparison_ready'])
        self.assertEqual([g['data']['documents'] for g in result['domain_groups']],[0,0])
