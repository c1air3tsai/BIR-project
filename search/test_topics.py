import io
import math
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import IntegrityError, transaction
from django.test import TestCase, SimpleTestCase
from django.urls import reverse
from .analysis import analyze, condition_tokens, fit_log_log
from .forms import TopicImportForm, get_or_create_topic
from .indexer import index_file, build_index, find_duplicate_document
from .models import Document, Topic, ImportJob
from .pmc_client import fetch_pubmed_batch
from .topic_import import create_job, run_job


def xml(pmid, title=None, abstract='The patients improved. Treatment helps patients.', doi='', language='eng'):
    return f'<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><ArticleTitle>{title or "Article "+str(pmid)}</ArticleTitle><Language>{language}</Language><Journal><Title>Test Journal</Title><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal><Abstract><AbstractText>{abstract}</AbstractText></Abstract></Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">{doi}</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>'.encode()


class ZipfMathTests(SimpleTestCase):
    def test_exact_zipf_is_recovered(self):
        rows=[{'rank':i,'cf':120/i} for i in range(1,31)]
        fit=fit_log_log(rows)
        self.assertAlmostEqual(fit['exponent'],1)
        self.assertAlmostEqual(fit['intercept'],math.log(120))
        self.assertAlmostEqual(fit['r2'],1)
        self.assertAlmostEqual(fit['rmse'],0)

    def test_cf_df_idf_and_cumulative_conditions(self):
        result=analyze(['The patients improved.','The patient improved improved.'])
        self.assertIn('.',condition_tokens('GLP-1 improved. 0.05')['A'])
        self.assertEqual(condition_tokens('GLP-1 improved. 0.05')['A'], ['glp-1','improved','.','0.05'])
        self.assertEqual(condition_tokens('GLP-1 improved. 0.05')['B'],['glp-1','improved','0.05'])
        self.assertGreater(result['A']['tokens'],result['B']['tokens'])
        rows={r['term']:r for r in result['D']['rows']}
        self.assertEqual(rows['improv']['cf'],3)
        self.assertEqual(rows['improv']['df'],2)
        self.assertEqual(rows['improv']['idf'],0)
        self.assertEqual(rows['patient']['cf'],2)
        self.assertNotIn('the',rows)
        self.assertEqual(result['C']['tokens'],result['D']['tokens'])
        self.assertGreater(result['C']['vocabulary'],result['D']['vocabulary'])

    def test_empty_and_flat_distribution(self):
        self.assertEqual(analyze([])['D']['vocabulary'],0)
        self.assertIsNone(fit_log_log([{'rank':1,'cf':1}])['slope'])
        self.assertIsNone(fit_log_log([{'rank':1,'cf':1},{'rank':2,'cf':1}])['r2'])


class TopicFeaturesTests(TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.directory=Path(self.temp.name)
        self.settings_override=self.settings(BASE_DIR=self.directory)
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)
        self.addCleanup(self.temp.cleanup)
        cache.clear()

    def add_article(self,pmid,topics=(),abstract='The patients improved. Treatment helps patients.'):
        path=self.directory/'data'/'corpus'/f'{pmid}.xml'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(xml(pmid,abstract=abstract))
        doc=index_file(str(path));doc.topics.set(topics);return doc

    def test_multi_topic_union_and_empty_selection_all_on_every_page(self):
        a=get_or_create_topic('GLP-1');b=get_or_create_topic('diabetes')
        self.add_article(1,[a]);self.add_article(2,[a,b]);self.add_article(3,[b]);self.add_article(4)
        for name in ['articles','home','zipf']:
            params={'topic':[a.pk,b.pk]}
            if name=='home':params['q']='patients'
            response=self.client.get(reverse('search:'+name),params)
            self.assertEqual(response.status_code,200)
            if name=='zipf':count=response.context['selected']['documents']
            else:count=response.context['page_obj'].paginator.count
            self.assertEqual(count,3)
            response=self.client.get(reverse('search:'+name),{'q':'patients'} if name=='home' else {})
            if name=='zipf':count=response.context['selected']['documents']
            else:count=response.context['page_obj'].paginator.count
            self.assertEqual(count,4)
        response=self.client.get(reverse('search:zipf'),{'topic':[a.pk,b.pk]})
        row=next(r for r in response.context['selected']['rows'] if r['term']=='patient')
        self.assertEqual(row['df'],3)

    def test_twenty_articles_per_page_and_filters_survive_pagination(self):
        a=get_or_create_topic('GLP-1')
        for i in range(22):self.add_article(100+i,[a])
        for name in ['articles','home']:
            response=self.client.get(reverse('search:'+name),{'topic':a.pk,'q':'patients'})
            self.assertEqual(len(response.context['page_obj']),20)
            self.assertContains(response,'topic='+str(a.pk))
            response=self.client.get(reverse('search:'+name),{'topic':a.pk,'q':'patients','page':2})
            self.assertEqual(len(response.context['page_obj']),2)

    def test_manual_duplicate_reuses_article_and_adds_another_topic(self):
        doc=self.add_article(22)
        response=self.client.post(reverse('search:import'),{
            'action':'upload','topic':'Diabetes','files':[SimpleUploadedFile('new.xml',xml(22))]})
        self.assertEqual(response.status_code,302)
        self.assertEqual(Document.objects.count(),1)
        self.assertEqual(doc.topics.get().name,'Diabetes')

    def test_same_filename_with_different_pmid_is_not_a_duplicate(self):
        doc=self.add_article(30)
        self.assertIsNone(find_duplicate_document('Article 31',{'pmid':'31'},doc.source_file))

    def test_rebuilding_index_preserves_ids_and_topics_and_duplicate_files(self):
        a=get_or_create_topic('GLP-1');doc=self.add_article(40,[a])
        (self.directory/'data'/'corpus'/'copy.xml').write_bytes(xml(40))
        build_index(str(self.directory/'data'/'corpus'))
        self.assertEqual(Document.objects.count(),1)
        self.assertEqual(Document.objects.get().pk,doc.pk)
        self.assertEqual(Document.objects.get().topics.get().pk,a.pk)
        self.assertTrue(doc.postings.exists())

    def test_scoped_export_contains_all_conditions_and_only_selected_articles(self):
        a=get_or_create_topic('GLP-1');self.add_article(50,[a]);self.add_article(51)
        response=self.client.get(reverse('search:zipf_export'),{'topic':a.pk})
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            self.assertIn('D_terms.csv',z.namelist())
            self.assertIn('methodology.json',z.namelist())
            corpus=z.read('corpus.csv').decode('utf-8-sig')
            self.assertIn('Article 50',corpus)
            self.assertNotIn('Article 51',corpus)

    def test_term_link_uses_exact_condition_and_preserves_abstract_matches(self):
        doc=self.add_article(52,abstract='Patients patient. Patients improved.')
        response=self.client.get(reverse('search:home'),{'analysis_term':'patient','condition':'B'})
        self.assertEqual(response.context['page_obj'][0]['keyword_count'],1)
        response=self.client.get(reverse('search:document_detail',args=[doc.pk]),{'analysis_term':'patient','condition':'B'})
        self.assertEqual(response.context['keyword_count'],1)
        self.assertEqual(response.context['abstract_match_count'],1)
        response=self.client.get(reverse('search:document_detail',args=[doc.pk]),{'analysis_term':'patient','condition':'D'})
        self.assertEqual(response.context['abstract_match_count'],3)

    def test_form_limits_query_and_quantities(self):
        for count in [0,1001]:self.assertFalse(TopicImportForm({'query':'GLP-1','count':count}).is_valid())
        self.assertTrue(TopicImportForm({'query':'GLP-1','count':1000}).is_valid())
        self.assertEqual(get_or_create_topic('  GLP-1 ').pk,get_or_create_topic('glp-1').pk)

    def test_worker_deduplicates_and_reuses_other_topic_articles(self):
        existing=self.add_article(60)
        a=get_or_create_topic('GLP-1');b=get_or_create_topic('diabetes');existing.topics.add(b)
        job=create_job('GLP-1',2)
        with patch('search.topic_import.search_pubmed',return_value=(['60','61'],2)),patch('search.topic_import.fetch_pubmed_batch',return_value=[('60',xml(60)),('61',xml(61))]):
            run_job(job.pk)
        job.refresh_from_db()
        self.assertEqual(job.status,'completed')
        self.assertEqual((job.imported,job.linked),(1,1))
        self.assertEqual(Document.objects.count(),2)
        self.assertEqual(a.documents.count(),2)
        self.assertEqual(existing.topics.count(),2)

    def test_worker_respects_stop_request_during_fetch_and_keeps_articles(self):
        job=create_job('GLP-1',2)
        def cancel_and_fetch(ids):
            ImportJob.objects.filter(pk=job.pk).update(cancel_requested=True)
            return [('70',xml(70))]
        with patch('search.topic_import.search_pubmed',return_value=(['70'],1)),patch('search.topic_import.fetch_pubmed_batch',side_effect=cancel_and_fetch):
            run_job(job.pk)
        job.refresh_from_db();self.assertEqual(job.status,'cancelled');self.assertTrue(job.cancel_requested)
        self.assertEqual(Document.objects.count(),0)

    def test_partial_and_network_failure_keep_prior_work(self):
        job=create_job('GLP-1',5)
        with patch('search.topic_import.search_pubmed',return_value=(['80'],1)),patch('search.topic_import.fetch_pubmed_batch',return_value=[('80',xml(80))]):run_job(job.pk)
        job.refresh_from_db();self.assertEqual(job.status,'partial');self.assertEqual(job.added,1)
        later=create_job('diabetes',1)
        with patch('search.topic_import.search_pubmed',side_effect=RuntimeError('No network')):
            with self.assertRaises(RuntimeError):run_job(later.pk)
        later.refresh_from_db();self.assertEqual(later.status,'failed');self.assertEqual(Document.objects.count(),1)

    def test_only_one_active_import(self):
        create_job('GLP-1',1)
        with self.assertRaises(IntegrityError),transaction.atomic():create_job('diabetes',1)

    def test_failed_record_rolls_back_article_file_and_job_count_together(self):
        from .topic_import import _persist
        job=create_job('GLP-1',1)
        def fail_on_added(current,**fields):
            if fields.get('message','').startswith('Adding to'):
                raise RuntimeError('Simulated interrupted write')
            return _persist(current,**fields)
        with patch('search.topic_import.search_pubmed',return_value=(['81'],1)),patch('search.topic_import.fetch_pubmed_batch',return_value=[('81',xml(81))]),patch('search.topic_import._persist',side_effect=fail_on_added):
            with self.assertRaises(RuntimeError):run_job(job.pk)
        job.refresh_from_db()
        self.assertEqual(job.status,'failed')
        self.assertEqual(job.added,0)
        self.assertEqual(Document.objects.count(),0)
        self.assertEqual(list((self.directory/'data'/'corpus').glob('*.xml')),[])

    def test_blank_abstract_is_excluded_from_n(self):
        Document.objects.create(title='Empty',abstract='   ',raw_text='',source_file='empty.xml')
        self.add_article(90)
        response=self.client.get(reverse('search:zipf'))
        self.assertEqual(response.context['selected']['documents'],1)
        self.assertEqual(response.context['excluded_count'],1)

    def test_delete_all_articles_in_one_topic(self):
        a=get_or_create_topic('GLP-1');b=get_or_create_topic('diabetes')
        first=self.add_article(91,[a])
        shared=self.add_article(92,[a,b])
        kept=self.add_article(93,[b])
        response=self.client.post(reverse('search:delete_topic_documents'),{'topic':a.pk})
        self.assertRedirects(response,reverse('search:articles'),fetch_redirect_response=False)
        self.assertFalse(Document.objects.filter(pk__in=[first.pk,shared.pk]).exists())
        self.assertTrue(Document.objects.filter(pk=kept.pk).exists())
        self.assertFalse((self.directory/'data'/'corpus'/'91.xml').exists())
        self.assertFalse((self.directory/'data'/'corpus'/'92.xml').exists())

    def test_zipf_shows_top_fifty_as_two_twenty_five_term_columns(self):
        abstract=' '.join(f'term{i}' for i in range(60))
        self.add_article(94,abstract=abstract)
        response=self.client.get(reverse('search:zipf'),{'condition':'B'})
        self.assertEqual([len(column) for column in response.context['term_columns']],[25,25])
        self.assertEqual(response.context['term_page'].paginator.per_page,50)

    def test_batched_xml_response_filters_non_english(self):
        raw=b'<PubmedArticleSet>'+xml(100)[18:-19]+xml(101,language='spa')[18:-19]+b'</PubmedArticleSet>'
        # Use real XML trees to avoid depending on wrapper byte lengths.
        from xml.etree import ElementTree as ET
        root=ET.Element('PubmedArticleSet')
        root.append(ET.fromstring(xml(100))[0]);root.append(ET.fromstring(xml(101,language='spa'))[0])
        with patch('search.pmc_client._get',return_value=ET.tostring(root)):
            self.assertEqual([pmid for pmid,_ in fetch_pubmed_batch(['100','101'])],['100'])

    def test_temporary_source_failure_after_100_articles_keeps_importing(self):
        from .pmc_client import NCBITransientError
        ids = [str(10000 + i) for i in range(155)]
        job = create_job('Another topic', 150)
        failed_once = False
        def fetch(batch):
            nonlocal failed_once
            if batch[0] == ids[100] and not failed_once:
                failed_once = True
                raise NCBITransientError('Temporary connection reset')
            return [(pmid, xml(pmid)) for pmid in batch]
        with patch('search.topic_import.search_pubmed', return_value=(ids, len(ids))), patch('search.topic_import.fetch_pubmed_batch', side_effect=fetch), patch('search.topic_import.time.sleep'):
            run_job(job.pk)
        job.refresh_from_db()
        self.assertTrue(failed_once)
        self.assertEqual((job.status, job.imported), ('completed', 150))
        self.assertEqual(Document.objects.count(), 150)

    def test_malformed_single_article_does_not_stop_other_articles(self):
        job = create_job('Another topic', 2)
        with patch('search.topic_import.search_pubmed', return_value=(['1','2','3'],3)), patch('search.topic_import.fetch_pubmed_batch', return_value=[('1',b'<broken'),('2',xml(2)),('3',xml(3))]):
            run_job(job.pk)
        job.refresh_from_db()
        self.assertEqual((job.status,job.imported,job.skipped),('completed',2,1))

    def test_stop_during_source_retry_remains_cancelled(self):
        from .pmc_client import NCBITransientError
        job=create_job('Another topic',1)
        def failure(batch):
            ImportJob.objects.filter(pk=job.pk).update(cancel_requested=True)
            raise NCBITransientError('Temporary connection reset')
        with patch('search.topic_import.search_pubmed', return_value=(['1'],1)), patch('search.topic_import.fetch_pubmed_batch',side_effect=failure),patch('search.topic_import.time.sleep'):
            run_job(job.pk)
        job.refresh_from_db()
        self.assertEqual(job.status,'cancelled')
