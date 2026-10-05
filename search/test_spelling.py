import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from django.test import TestCase, SimpleTestCase
from django.urls import reverse
from django.core.cache import cache
from .models import Document
from .forms import get_or_create_topic
from .indexer import index_file
from .spelling import suggest_query, bounded_distance
from .distance import edit_distance, edit_matrix
from .test_topics import xml


class BoundedDistanceTests(SimpleTestCase):
    def test_banded_distance_matches_full_algorithm(self):
        words=['','cancer','camcer','caner','cancers','cancerous','sitting','kitten','abcdefgh','abcedfgh','école']
        for left in words:
            for right in words:
                for limit in (1,2,3,4):
                    self.assertEqual(min(bounded_distance(left,right,limit),limit+1),min(edit_distance(left,right),limit+1),(left,right,limit))


class SpellingSearchTests(TestCase):
    def setUp(self):
        cache.clear()
        self.temp=tempfile.TemporaryDirectory();self.directory=Path(self.temp.name)
        override=self.settings(BASE_DIR=self.directory);override.enable();self.addCleanup(override.disable);self.addCleanup(self.temp.cleanup)
        self.med=get_or_create_topic('Medical');self.cs=get_or_create_topic('CS')
        self.add(1,'Cancer treatment benefits patients. Cancer risk matters.',self.med,year='2025')
        self.add(2,'Cancer treatment improves survival.',self.med,year='2025')
        self.add(3,'Canker research describes lesions.',self.med,year='2024')
        self.add(4,'Computer camera models improve treatment algorithms.',self.cs,year='2025')

    def add(self,pmid,text,topic,year='2025'):
        path=self.directory/f'{pmid}.xml'
        path.write_bytes(xml(pmid,abstract=text).replace(b'<Year>2025</Year>',f'<Year>{year}</Year>'.encode()))
        doc=index_file(str(path));doc.topics.add(topic);return doc

    def test_no_results_shows_real_correction_without_automatic_search(self):
        response=self.client.get(reverse('search:home'),{'q':'camcer','topic':self.med.pk,'year':'2025','page':4})
        self.assertEqual(response.context['query'],'camcer')
        self.assertEqual(response.context['page_obj'].paginator.count,0)
        suggestions=response.context['spelling_suggestions'];self.assertEqual(suggestions[0]['query'],'cancer')
        self.assertContains(response,'Did you mean?')
        from html.parser import HTMLParser
        class Links(HTMLParser):
            links=[]
            def handle_starttag(self,tag,attrs):
                a=dict(attrs)
                if tag=='a' and a.get('class')=='spelling-option':self.links.append(a['href'])
        parser=Links();parser.feed(response.content.decode());link=parser.links[0]
        query=parse_qs(urlparse(link).query)
        self.assertEqual(query['q'],['cancer']);self.assertEqual(query['topic'],[str(self.med.pk)]);self.assertEqual(query['year'],['2025']);self.assertNotIn('page',query)
        corrected=self.client.get(link);self.assertEqual(corrected.context['query'],'cancer');self.assertEqual(corrected.context['page_obj'].paginator.count,2)
        self.assertEqual(corrected.context['spelling_suggestions'],[])

    def test_partial_match_keeps_results_and_replaces_only_suspected_word(self):
        response=self.client.get(reverse('search:home'),{'q':'CAMCER treatment','topic':self.med.pk})
        self.assertEqual(response.context['query'],'CAMCER treatment');self.assertGreater(response.context['page_obj'].paginator.count,0)
        self.assertEqual(response.context['spelling_suggestions'][0]['query'],'CANCER treatment')

    def test_topics_and_year_restrict_candidates(self):
        response=self.client.get(reverse('search:home'),{'q':'camcer','topic':self.cs.pk})
        self.assertEqual(response.context['spelling_suggestions'][0]['query'],'camera')
        self.assertEqual(response.context['spelling_suggestions'][0]['distance'],2)
        response=self.client.get(reverse('search:home'),{'q':'caner','topic':self.med.pk,'year':'2024'})
        self.assertEqual([r['query'] for r in response.context['spelling_suggestions']],['canker'])

    def test_existing_rare_words_inflections_stopwords_and_ids_are_not_changed(self):
        for query in ['patients','treatments','the','GLP-1','zzzzzz','cat','p53']:
            self.assertEqual(suggest_query(query,Document.objects.all()),[],query)

    def test_search_always_suggests_words_one_and_two_edits_away_with_distance(self):
        two=suggest_query('canr',Document.objects.filter(topics=self.med))
        self.assertEqual((two[0]['query'],two[0]['distance']),('cancer',2))
        successful=self.client.get(reverse('search:home'),{'q':'canker','topic':self.med.pk})
        self.assertGreater(successful.context['page_obj'].paginator.count,0)
        suggestion=next(row for row in successful.context['spelling_suggestions'] if row['query']=='cancer')
        self.assertEqual(suggestion['distance'],2)
        self.assertContains(successful,'Distance 2')

    def test_vocabulary_refreshes_after_new_article_and_multiple_typo_correction(self):
        before=suggest_query('neoplasmm',Document.objects.all());self.assertEqual(before,[])
        self.add(5,'Neoplasm treatment helps patients.',self.med)
        after=suggest_query('neoplasmm',Document.objects.all());self.assertEqual(after[0]['query'],'neoplasm')
        suggestions=suggest_query('camcer treatmant',Document.objects.all())
        self.assertIn('cancer treatmant',[row['query'] for row in suggestions])
        self.assertIn('camcer treatment',[row['query'] for row in suggestions])

    def test_candidates_rank_by_distance_then_document_frequency(self):
        self.add(5,'Caster opinions.',self.med)
        suggestions=suggest_query('caner',Document.objects.all())
        self.assertEqual(suggestions[0]['query'],'cancer')
        self.assertEqual(len(set(r['query'] for r in suggestions)),len(suggestions))

    def test_exact_zipf_term_search_does_not_offer_spelling_replacements(self):
        response=self.client.get(reverse('search:home'),{'analysis_term':'camcer','condition':'D'})
        self.assertEqual(response.context['spelling_suggestions'],[])

    def test_hyphenated_biomedical_identifier_is_expanded_both_ways(self):
        self.add(5,'GLP-1 therapy improves metabolic outcomes.',self.med)
        self.add(6,'GLP1 receptor research continues.',self.med)

        compact=self.client.get(reverse('search:home'),{'q':'glp1','topic':self.med.pk})
        self.assertEqual(compact.context['page_obj'].paginator.count,2)
        self.assertEqual(compact.context['search_variants'],[
            {'query_term':'glp1','variants':['glp-1']},
        ])
        self.assertContains(compact,'<mark class="search-highlight">GLP-1</mark>',html=True)

        hyphenated=self.client.get(reverse('search:home'),{'q':'GLP-1','topic':self.med.pk})
        self.assertEqual(hyphenated.context['page_obj'].paginator.count,2)
        self.assertEqual(hyphenated.context['search_variants'],[
            {'query_term':'glp-1','variants':['glp1']},
        ])

        typo=self.client.get(reverse('search:home'),{'q':'glp11','topic':self.med.pk})
        self.assertEqual(typo.context['page_obj'].paginator.count,0)
        suggestions={(row['query'],row['distance']) for row in typo.context['spelling_suggestions']}
        self.assertIn(('glp1',1),suggestions)
        self.assertIn(('glp-1',1),suggestions)

    def test_variant_expansion_does_not_fold_apostrophes_or_unrelated_words(self):
        self.add(5,"Can't recover re-cover examples.",self.med)
        apostrophe=self.client.get(reverse('search:home'),{'q':'cant','topic':self.med.pk})
        self.assertEqual(apostrophe.context['page_obj'].paginator.count,0)
        self.assertEqual(apostrophe.context['search_variants'],[])

    def test_explicit_equivalent_terms_are_scored_only_once(self):
        self.add(5,'GLP-1 cancer study.',self.med)
        response=self.client.get(reverse('search:home'),{
            'q':'glp1 glp-1 cancer','topic':self.med.pk,
        })
        self.assertEqual(response.context['page_obj'].paginator.count,3)
        self.assertEqual(response.context['search_variants'],[])

    def test_text_matching_groups_all_candidates_and_builds_dp_grid(self):
        response=self.client.get(reverse('search:distance'),{'term':'canr','topic':self.med.pk,'candidate':'cancer'})
        self.assertEqual(response.status_code,200)
        groups={group['distance']:group for group in response.context['candidate_groups']}
        self.assertIn('cancer',[row['term'] for row in groups[2]['rows']])
        self.assertEqual(response.context['selected_candidate']['term'],'cancer')
        self.assertEqual(response.context['matrix']['distance'],2)
        self.assertNotContains(response,'What the algorithm compares')

    def test_dp_matrix_matches_rolling_distance(self):
        matrix=edit_matrix('kitten','sitting')
        self.assertEqual(matrix['distance'],edit_distance('kitten','sitting'))
        self.assertTrue(matrix['rows'][-1]['cells'][-1]['on_path'])
