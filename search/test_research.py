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
from django.core.management import call_command
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
        override=self.settings(BASE_DIR=self.directory,EMBEDDING_CACHE_DIR=self.directory/'data'/'embedding_cache');override.enable();self.addCleanup(override.disable);self.addCleanup(self.temp.cleanup)
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

    def test_part2_dual_models_loss_tsne_and_parameter_experiment(self):
        records=embeddings.corpus_records(Document.objects.all());params=self.params(window=2)
        models=embeddings.train_both(records,params)
        self.assertEqual(set(models),{'cbow','skipgram'})
        for model in models.values():
            metadata=model.bir_metadata
            self.assertEqual(len(metadata['losses']),params['epochs'])
            self.assertTrue(all(row['loss']>=0 for row in metadata['losses']))
            self.assertEqual(metadata['parameters']['seed'],42)
            self.assertEqual(metadata['parameters']['workers'],1)
            self.assertTrue(metadata['sample_preview']['context'])
        targets=embeddings.default_targets(models)
        self.assertEqual(embeddings.parse_targets('',models,params['condition']),targets)
        self.assertEqual(embeddings.parse_targets(targets[0],models,params['condition']),[targets[0]])
        self.assertEqual(embeddings.parse_targets('not-in-the-vocabulary',models,params['condition']),[])
        comparison=embeddings.neighbor_comparison(models,targets[:2])
        self.assertEqual(len(comparison),2)
        self.assertIn('jaccard',comparison[0])
        stats=embeddings.corpus_statistics(records,params['condition'])
        self.assertTrue(embeddings.semantic_pairs(models))
        self.assertTrue(embeddings.rare_word_analysis(models,stats['frequencies']))
        visualizations={name:embeddings.semantic_visualizations(model,targets) for name,model in models.items()}
        for data in visualizations.values():
            self.assertEqual([row['term'] for row in data['pca']['points']],[row['term'] for row in data['tsne']['points']])
            self.assertLess(data['tsne']['perplexity'],len(targets))
        kind,label,configs,experiment_models=embeddings.train_experiment(records,params,'dimensions')
        experiment=embeddings.experiment_result(label,kind,configs,experiment_models,targets[0])
        self.assertIsNotNone(embeddings.load_experiment_result(records,params,'dimensions'))
        self.assertEqual([row['value'] for row in experiment['rows']],[50,50,100,100])
        self.assertEqual([row['architecture'] for row in experiment['rows']],['cbow','skipgram','cbow','skipgram'])
        self.assertIn('Only Embedding dimension changed',experiment['description'])
        differing=[key for key in configs[0] if configs[0][key]!=configs[1][key]]
        self.assertEqual(differing,['dimensions'])
        custom={**params,'compare_window_a':1,'compare_window_b':4,'compare_dimensions_a':25,'compare_dimensions_b':75,'compare_min_count_a':1,'compare_min_count_b':3}
        self.assertEqual([config['window'] for config in embeddings.experiment_config(custom,'window')[2]],[1,4])
        self.assertEqual([config['dimensions'] for config in embeddings.experiment_config(custom,'dimensions')[2]],[25,75])
        self.assertEqual([config['min_count'] for config in embeddings.experiment_config(custom,'min_count')[2]],[1,3])

    def test_persistent_pair_cache_manifest_artifacts_and_cache_hit(self):
        selection=embeddings.select_corpus(Document.objects.all(),{'corpus_preset':'custom','medical_count':3,'cs_count':3})
        records=selection['records'];params=self.params()
        key=embeddings.shared_cache_key(records,params)
        self.assertEqual(key,embeddings.shared_cache_key(list(reversed(records)),params))
        changed=[*records[:-1],(records[-1][0],records[-1][1]+' changed',records[-1][2])]
        self.assertNotEqual(key,embeddings.shared_cache_key(changed,params))
        models=embeddings.train_both(records,params)
        artifacts=embeddings.build_global_artifacts(records,params,models)
        directory=embeddings.cache_directory(records,params)
        for name in ['manifest.json','cbow.model','skipgram.model','cbow_loss.json','skipgram_loss.json','semantic_metadata.json','rare_word_metadata.json','cbow_global_pca.json','cbow_global_tsne.json','skipgram_global_pca.json','skipgram_global_tsne.json']:
            self.assertTrue((directory/name).exists(),name)
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        for field in ['created_at','document_count','document_ids','document_ids_hash','content_hash','model_settings','software','cache_version']:
            self.assertIn(field,manifest)
        self.assertEqual(set(manifest['models']),{'cbow','skipgram'})
        reloaded=embeddings.load_both(records,params)
        self.assertTrue(all(reloaded.values()))
        self.assertEqual(reloaded['cbow'].bir_metadata['document_ids'],reloaded['skipgram'].bir_metadata['document_ids'])
        self.assertEqual([p['term'] for p in artifacts['visualizations']['cbow']['pca']['points']],[p['term'] for p in artifacts['visualizations']['skipgram']['tsne']['points']])
        import numpy as np
        norms=np.linalg.norm(embeddings.normalized_vectors(reloaded['cbow'],artifacts['words']),axis=1)
        np.testing.assert_allclose(norms,1,atol=1e-6)
        for visual in artifacts['visualizations'].values():
            self.assertLess(visual['tsne']['perplexity'],visual['tsne']['number_of_points'])

    def test_pretrained_preset_hit_and_export_never_retrain(self):
        first=embeddings.precompute_preset(Document.objects.all(),'medical')
        self.assertFalse(first['cache_hit'])
        with patch('search.embeddings.train_both') as train_both:
            second=embeddings.precompute_preset(Document.objects.all(),'medical')
            self.assertTrue(second['cache_hit']);train_both.assert_not_called()
        params=embeddings.DEFAULTS
        page=self.client.get(reverse('search:word2vec'),{**params,'corpus_preset':'medical'})
        self.assertEqual(page.status_code,200)
        self.assertEqual(page.context['cache_status']['state'],'pretrained')
        self.assertNotIn('local_visualizations',page.context)
        self.assertNotContains(page,'id="local-map"')
        target=page.context['targets'][0]
        local_page=self.client.get(reverse('search:word2vec'),{**params,'corpus_preset':'medical','local_word':target})
        self.assertEqual(set(local_page.context['global_visualizations']),{'cbow','skipgram'})
        for visual in local_page.context['global_visualizations'].values():
            for method in ['pca','tsne']:
                chart=visual[method]
                self.assertNotIn('label_mode',chart)
                highlights=[point['highlight'] for point in chart['points']]
                self.assertEqual(highlights.count('target'),1)
                self.assertIn('neighbor',highlights)
                self.assertLessEqual(highlights.count('neighbor'),10)
        with patch('search.embeddings.train_both') as train_both:
            response=self.client.get(reverse('search:word2vec_export'),{**params,'corpus_preset':'medical'})
            self.assertEqual(response.status_code,200);train_both.assert_not_called()
        manifest=second['manifest']
        self.assertEqual(manifest['preset_name'],'glp1_1000')
        self.assertEqual(manifest['domain_counts'],{'medical':3,'cs':0})

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
        response=self.client.post(reverse('search:word2vec'),{**self.params(),'corpus_preset':'custom','medical_count':3,'cs_count':3,'local_word':'learning','action':'train_models'})
        self.assertEqual(response.status_code,302)
        self.assertTrue(response.url.endswith('#training-setup'))
        response=self.client.get(response.url);self.assertTrue(response.context['both_ready']);self.assertEqual(response.context['corpus_stats']['documents'],6)
        self.assertContains(response,'id="global-cbow-pca"');self.assertContains(response,'id="global-skipgram-tsne"')
        self.assertContains(response,'class="panel embedding-setup-panel"')
        self.assertContains(response,'action="/analysis/word2vec/#global-map"')
        self.assertNotContains(response,'id="local-map"')
        self.assertEqual(response.content.count(b'class="parameter-result-card"'),3)
        comparison=self.client.post(reverse('search:word2vec'),{**self.params(),'corpus_preset':'custom','medical_count':3,'cs_count':3,'targets':'learning','experiment':'all','action':'experiment','compare_window_a':1,'compare_window_b':3,'compare_dimensions_a':25,'compare_dimensions_b':50,'compare_min_count_a':1,'compare_min_count_b':2})
        self.assertTrue(comparison.url.endswith('#parameter-comparison'))
        comparison_page=self.client.get(comparison.url)
        self.assertEqual([card['values'] for card in comparison_page.context['experiment_cards']],[[1,3],[25,50],[1,2]])
        self.assertTrue(all(card['result'] for card in comparison_page.context['experiment_cards']))
        self.assertContains(comparison_page,'name="compare_window_a"')
        exported=self.client.get(reverse('search:word2vec_export'),{**self.params(),'corpus_preset':'custom','medical_count':3,'cs_count':3})
        with zipfile.ZipFile(io.BytesIO(exported.content)) as z:
            self.assertIn('cbow_vectors.txt',z.namelist());self.assertIn('skipgram_tsne.json',z.namelist());self.assertTrue(z.read('cbow_vectors.txt').splitlines()[0].endswith(b' 50'))
        for name,required in [('domains_export','domains_summary.csv'),('research_export','tfidf.csv')]:
            response=self.client.get(reverse('search:'+name));self.assertEqual(response.status_code,200)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:self.assertIn(required,z.namelist())

    def test_no_get_training_and_insufficient_vocabulary(self):
        with patch('search.embeddings.train') as train:
            self.client.get(reverse('search:word2vec'));train.assert_not_called()
        with self.assertRaises(ValueError):embeddings.train([(1,'X.')],self.params(min_count=5))

    def test_corpus_presets_preprocessing_and_shared_document_ids(self):
        default=embeddings.select_corpus(Document.objects.all(),{})
        balanced=embeddings.select_corpus(Document.objects.all(),{'corpus_preset':'balanced'})
        self.assertEqual(default['actual'],{'medical':3,'cs':0})
        self.assertEqual(balanced['requested'],{'medical':3,'cs':3})
        medical=embeddings.select_corpus(Document.objects.all(),{'corpus_preset':'medical'})
        cs=embeddings.select_corpus(Document.objects.all(),{'corpus_preset':'cs'})
        all_docs=embeddings.select_corpus(Document.objects.all(),{'corpus_preset':'all'})
        self.assertEqual((len(medical['records']),len(cs['records']),len(all_docs['records'])),(3,3,6))
        custom=embeddings.select_corpus(Document.objects.all(),{'corpus_preset':'custom','medical_count':'2','cs_count':'2'})
        repeated=embeddings.select_corpus(Document.objects.all(),{'corpus_preset':'custom','medical_count':'2','cs_count':'2'})
        self.assertEqual(custom['document_ids'],repeated['document_ids'])
        models=embeddings.train_both(custom['records'],self.params())
        self.assertEqual(models['cbow'].bir_metadata['document_ids'],models['skipgram'].bir_metadata['document_ids'])
        self.assertEqual(models['cbow'].vector_size,50);self.assertEqual(models['skipgram'].vector_size,50)
        self.assertIn('studies',embeddings.condition_tokens('Studies computing effects')['C'])
        self.assertIn('studi',embeddings.condition_tokens('Studies computing effects')['D'])

    def test_keyed_bilingual_word_embedding_ui(self):
        response=self.client.get(reverse('search:word2vec'))
        self.assertContains(response,'data-i18n-key="we.corpus.title"')
        template=(Path(__file__).parent/'templates'/'search'/'word2vec.html').read_text(encoding='utf-8')
        javascript=(Path(__file__).parent/'static'/'search'/'i18n.js').read_text(encoding='utf-8')
        import re
        keys=set(re.findall(r'data-i18n(?:-aria)?-key="([^"]+)"',template))
        self.assertFalse([key for key in keys if '{' not in key and f"'{key}'" not in javascript])
        for key in ['we.field.condition','we.field.dimensions','we.field.window','we.field.min_count','we.field.epochs','we.condition.B','we.condition.C','we.condition.D']:
            self.assertIn(f"'{key}'",javascript)
        self.assertNotContains(response,'RQ1');self.assertNotContains(response,'Change exactly one setting')
        self.assertIn('1–5 ·',template);self.assertIn('class="step-label"',template)
        for marker in ['setup-grid','setup-group','fixed-settings-details','parameter-input-grid','compare_{{ card.kind }}_a','id="parameter-comparison"']:
            self.assertIn(marker,template)
        for marker in ['map-neighbor-results','we.global.neighbor.list','Top-10']:
            self.assertIn(marker,template)
        projection=(Path(__file__).parent/'static'/'search'/'research.js').read_text(encoding='utf-8')
        for marker in ['projection-label','projection-tooltip','labelBudget','requestAnimationFrame','cellSize','safelyRender','finally','data-local-form','data-local-status']:
            self.assertIn(marker,projection)
        self.assertIn("url.hash='training-setup'",projection)
        for marker in ["point.highlight==='target'","point.highlight==='neighbor'","Math.max(10,rawRadius*1.6)","is-target","is-neighbor","#2e9d68"]:
            self.assertIn(marker,projection)
        self.assertNotIn('leader:add',projection)
        for marker in ['IntersectionObserver','requestIdleCallback','createDocumentFragment','dataset.renderState']:
            self.assertIn(marker,projection)
        self.assertNotIn('getBBox(',projection);self.assertNotIn('getBoundingClientRect()',projection.split('placeLabels=')[1].split('};',1)[0])
        self.assertIn('observer.disconnect()',javascript)
        self.assertIn('root.nodeType!==Node.ELEMENT_NODE&&root.nodeType!==Node.DOCUMENT_NODE',javascript)
        self.assertNotIn('const scope=root.querySelectorAll?root:document',javascript)

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


class WordEmbeddingCorpusScaleTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        documents=[]
        for i in range(1000):
            documents.append(Document(title=f'Medical {i}',source_file=f'med-{i}.xml',abstract='Medical treatment studies effects.',raw_text='',pmid=str(100000+i)))
        for i in range(500):
            documents.append(Document(title=f'CS {i}',source_file=f'cs-{i}.xml',abstract='Computer retrieval algorithms compute results.',raw_text='',arxiv_id=f'2601.{i:05d}'))
        Document.objects.bulk_create(documents,batch_size=500)

    def test_balanced_custom_medical_cs_and_all_1500(self):
        medical_default=embeddings.select_corpus(Document.objects.all(),{})
        balanced=embeddings.select_corpus(Document.objects.all(),{'corpus_preset':'balanced'})
        custom=embeddings.select_corpus(Document.objects.all(),{'corpus_preset':'custom','medical_count':500,'cs_count':500})
        medical=embeddings.select_corpus(Document.objects.all(),{'corpus_preset':'medical'})
        cs=embeddings.select_corpus(Document.objects.all(),{'corpus_preset':'cs'})
        all_docs=embeddings.select_corpus(Document.objects.all(),{'corpus_preset':'all'})
        self.assertEqual(medical_default['actual'],{'medical':1000,'cs':0})
        self.assertEqual(balanced['actual'],{'medical':500,'cs':500})
        self.assertEqual(custom['document_ids'],balanced['document_ids'])
        self.assertEqual((len(medical['records']),len(cs['records']),len(all_docs['records'])),(1000,500,1500))
        self.assertEqual(len(set(all_docs['document_ids'])),1500)
