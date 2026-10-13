import csv
import io
import json
import logging
import time
import zipfile
from collections import Counter
from urllib.parse import urlencode
from types import SimpleNamespace
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import HttpResponse, Http404
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from .models import Document
from .topic_filters import topic_context, filter_documents
from .analysis import CONDITIONS, analyze_queryset, condition_tokens
from . import embeddings
from .domain_analysis import compare_domains
from .research_report import report, markdown
from .distance import normalize, edit_matrix
from .spelling import bounded_distance

logger = logging.getLogger(__name__)

def csv_text(header,rows):
    stream=io.StringIO(newline='');writer=csv.writer(stream)
    writer.writerow(header)
    for row in rows: writer.writerow(["'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v for v in row])
    return '\ufeff'+stream.getvalue()

def download_zip(files,name):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        for path,content in files.items(): archive.writestr(path,content)
    response=HttpResponse(stream.getvalue(),content_type='application/zip')
    response['Content-Disposition']=f'attachment; filename="{name}.zip"'
    return response

def domains_view(request):
    return render(request,'search/domains.html',{**compare_domains(request.GET),'page_title':'Compare domains · BioMed IR','analysis_tab':'domains'})

def domains_export(request):
    result=compare_domains(request.GET)
    if not result['comparison_ready']:raise Http404('Select two domains containing abstracts first.')
    files={};groups=result['domain_groups'];condition=result['domain_condition']
    fields=['documents','tokens','vocabulary','avg_tokens']
    files['domains_summary.csv']=csv_text(['domain','topics','condition',*fields,'exponent','R2','RMSE'],([g['key'],g['name'],condition,*[g['data'][k] for k in fields],*[g['data']['fit'][k] for k in ['exponent','r2','rmse']]] for g in groups))
    for g in groups:
        files[g['key']+'_top10.csv']=csv_text(['term','CF','DF','IDF'],([r[k] for k in ['term','cf','df','idf']] for r in g['top10']))
        files[g['key']+'_corpus.csv']=csv_text(['local_id','PMID','PMCID','arXiv_ID','title','abstract'],Document.objects.filter(pk__in=g['document_ids']).order_by('pk').values_list('pk','pmid','pmcid','arxiv_id','title','abstract'))
        for key,data in g['analyses'].items():
            files[g['key']+'_'+key+'_terms.csv']=csv_text(['rank','term','CF','DF','IDF'],([r[k] for k in ['rank','term','cf','df','idf']] for r in data['rows']))
    files['methodology.json']=json.dumps({'condition':condition,'sample':result['sample'],'overlap_policy':result['overlap_mode'],'overlapping_documents':result['overlap_count'],'sampling':'All mode uses every eligible document independently; numeric mode balances to the smaller eligible set and takes ascending local IDs, up to the requested cap.','top10_jaccard':result['top10_jaccard'],'domains':[{'topics':g['name'],'document_ids':g['document_ids']} for g in groups]},ensure_ascii=False,indent=2)
    a,b=groups
    files['domain_observations.md']=f"# Domain comparison\n\n{a['name']}: {a['data']['documents']} abstracts; k={a['data']['fit']['exponent']}; R²={a['data']['fit']['r2']}.\n\n{b['name']}: {b['data']['documents']} abstracts; k={b['data']['fit']['exponent']}; R²={b['data']['fit']['r2']}.\n\nTop-10 Jaccard overlap={result['top10_jaccard']}. Compare vocabulary, top terms, and curve shape under identical preprocessing. Domain specificity may affect these descriptive results, but differences also reflect corpus size, source, sampling date and abstract style; this comparison does not isolate a causal domain effect."
    return download_zip(files,'domain_comparison')

@require_http_methods(['GET','POST'])
def _legacy_word2vec_view(request):
    filters=topic_context(SimpleNamespace(GET=request.POST) if request.method=='POST' else request)
    documents=filter_documents(Document.objects.all(),filters['selected_topics'])
    records=embeddings.corpus_records(documents)
    params=embeddings.parameters(request.POST if request.method=='POST' else request.GET)
    if request.method=='POST':
        try:
            action=request.POST.get('action','train')
            if action=='train_both':
                embeddings.train_both(records,params)
                messages.success(request,'CBOW and Skip-Gram were trained on the identical corpus and settings.')
            elif action=='experiment':
                embeddings.train_experiment(records,params,request.POST.get('experiment','window'))
                messages.success(request,'Parameter experiment completed. Only the selected parameter was changed.')
            else:
                embeddings.train(records,params)
                messages.success(request,'Word2Vec trained on your current abstracts. Explore similar words below.')
        except ValueError as exc:messages.error(request,str(exc))
        query=[(k,str(v)) for k,v in params.items()]+[('topic',t) for t in filters['selected_topics']]
        if request.POST.get('word'): query.append(('word',request.POST['word'][:120]))
        if request.POST.get('targets'): query.append(('targets',request.POST['targets'][:1000]))
        if request.POST.get('experiment'): query.append(('experiment',request.POST['experiment']))
        return redirect(reverse('search:word2vec')+'?'+urlencode(query))
    model=embeddings.load(records,params)
    word=request.GET.get('word','').strip()[:120]
    error=None;neighbors=[];projection={};overview_projection={};normalized='';visualizations={}
    if model is not None and word:
        try:
            normalized,neighbors=embeddings.neighbors(model,word,params['condition'])
            projection=embeddings.projection(model,[normalized,*[r['term'] for r in neighbors]])
        except ValueError as exc:error=str(exc)
    vocabulary=[]
    if model is not None:
        vocabulary=[{'term':w,'cf':model.wv.get_vecattr(w,'count')} for w in model.wv.index_to_key[:20]]
        overview_projection=embeddings.vocabulary_projection(model)
        visualizations=embeddings.semantic_visualizations(model,model.wv.index_to_key[:min(40,len(model.wv))])
    stats=embeddings.corpus_statistics(records,params['condition'])
    frequencies=stats.pop('frequencies');stats.pop('sentences_data')
    paired_models=embeddings.load_both(records,params)
    both_ready=all(paired_models.values())
    targets=[];neighbor_comparison=[];pair_rows=[];rare_rows=[];comparison_rows=[];research_report=[]
    targets_text=request.GET.get('targets','')[:1000]
    experiment=None
    if both_ready:
        targets=embeddings.parse_targets(targets_text,paired_models,params['condition'])
        targets_text=', '.join(targets)
        neighbor_comparison=embeddings.neighbor_comparison(paired_models,targets)
        pair_rows=embeddings.semantic_pairs(paired_models)
        rare_rows=embeddings.rare_word_analysis(paired_models,frequencies)
        comparison_rows=embeddings.comparison_table(paired_models,rare_rows)
    experiment_kind=request.GET.get('experiment','window')
    kind,label,configs,experiment_models=embeddings.load_experiment(records,params,experiment_kind)
    if all(experiment_models):
        experiment=embeddings.experiment_result(label,kind,configs,experiment_models,targets[0] if targets else normalized)
    if both_ready:
        research_report=embeddings.evidence_report(paired_models,neighbor_comparison,pair_rows,rare_rows,experiment,visualizations)
    loss_series=[]
    if both_ready:
        loss_series=[{'label':'CBOW','points':paired_models['cbow'].bir_metadata.get('losses',[])},{'label':'Skip-Gram','points':paired_models['skipgram'].bir_metadata.get('losses',[])}]
    elif model is not None:
        loss_series=[{'label':params['architecture'].replace('skipgram','Skip-Gram').replace('cbow','CBOW'),'points':model.bir_metadata.get('losses',[])}]
    training_sample=(paired_models['cbow'].bir_metadata.get('sample_preview') if both_ready else (model.bir_metadata.get('sample_preview') if model is not None else {}))
    parameter_choices=[{'name':name,'label':label,'options':[{'value':value,'label':{'skipgram':'Skip-Gram','cbow':'CBOW','A':'A · punctuation retained','B':'B · punctuation removed','C':'C · stopwords removed','D':'D · Porter stemming'}.get(value,str(value)),'selected':params[name]==value} for value in options]} for name,label,options in [('architecture','Architecture',['cbow','skipgram']),('condition','Preprocessing',['A','B','C','D']),('dimensions','Embedding dimension',[50,100]),('window','Context window',[2,5,10]),('min_count','Minimum word count',[1,2,5]),('epochs','Training epochs',[10,20,40])]]
    return render(request,'search/word2vec.html',{
        **filters,'analysis_tab':'word2vec','page_title':'Word2Vec · BioMed IR','params':params,
        'full_params':embeddings.full_parameters(params),'parameter_choices':parameter_choices,
        'model_metadata':model.bir_metadata if model is not None else None,'document_count':len(records),
        'corpus_stats':stats,'word':word,'normalized_word':normalized,'neighbors':neighbors,
        'embedding_error':error,'projection':projection,'overview_projection':overview_projection,
        'visualizations':visualizations,'model_vocabulary':vocabulary,'both_ready':both_ready,
        'paired_models':paired_models,'targets':targets,'targets_text':targets_text,
        'neighbor_comparison':neighbor_comparison,'semantic_pairs':pair_rows,'rare_rows':rare_rows,
        'comparison_rows':comparison_rows,'research_report':research_report,'experiment':experiment,
        'experiment_kind':experiment_kind,'experiment_choices':[{'value':key,'label':value[0]} for key,value in embeddings.EXPERIMENTS.items()],
        'loss_series':loss_series,'training_sample':training_sample,
    })

def _legacy_word2vec_export(request):
    filters=topic_context(request);records=embeddings.corpus_records(filter_documents(Document.objects.all(),filters['selected_topics']))
    params=embeddings.parameters(request.GET);model=embeddings.load(records,params)
    if model is None:raise Http404('Train this scope and parameter combination first.')
    header=f'{len(model.wv)} {model.vector_size}\n'
    vectors=header+'\n'.join(w+' '+' '.join(format(float(x),'.8g') for x in model.wv[w]) for w in model.wv.index_to_key)+'\n'
    files={'vectors.txt':vectors,'metadata.json':json.dumps(model.bir_metadata,ensure_ascii=False,indent=2),'corpus_ids.csv':csv_text(['local_id'],([pk] for pk,_ in records))}
    files['vocabulary_pca.json']=json.dumps(embeddings.vocabulary_projection(model),ensure_ascii=False,indent=2)
    files['training_loss.csv']=csv_text(['epoch','loss','cumulative_loss','seconds'],([r[k] for k in ['epoch','loss','cumulative_loss','seconds']] for r in model.bir_metadata.get('losses',[])))
    visualization=embeddings.semantic_visualizations(model,model.wv.index_to_key[:min(40,len(model.wv))])
    files['pca.json']=json.dumps(visualization['pca'],ensure_ascii=False,indent=2)
    files['tsne.json']=json.dumps(visualization['tsne'],ensure_ascii=False,indent=2)
    word=request.GET.get('word','')[:120]
    if word:
        try:
            normalized,rows=embeddings.neighbors(model,word,params['condition'])
            files['similar_words.csv']=csv_text(['query','neighbor','cosine_similarity'],([normalized,r['term'],r['similarity']] for r in rows))
            files['projection.json']=json.dumps(embeddings.projection(model,[normalized,*[r['term'] for r in rows]]))
        except ValueError:pass
    paired=embeddings.load_both(records,params)
    if all(paired.values()):
        stats=embeddings.corpus_statistics(records,params['condition'])
        targets=embeddings.parse_targets(request.GET.get('targets',''),paired,params['condition'])
        comparisons=embeddings.neighbor_comparison(paired,targets)
        pairs=embeddings.semantic_pairs(paired)
        rare=embeddings.rare_word_analysis(paired,stats['frequencies'])
        kind,label,configs,experiment_models=embeddings.load_experiment(records,params,request.GET.get('experiment','window'))
        experiment=embeddings.experiment_result(label,kind,configs,experiment_models,targets[0]) if all(experiment_models) else None
        evidence=embeddings.evidence_report(paired,comparisons,pairs,rare,experiment,visualization)
        files['cbow_vs_skipgram.csv']=csv_text(['property','CBOW','Skip-Gram'],embeddings.comparison_table(paired,rare))
        files['semantic_neighbors.csv']=csv_text(['target','architecture','rank','neighbor','cosine_similarity'],([row['target'],architecture,item['rank'],item['term'],item['similarity']] for row in comparisons for architecture in ['cbow','skipgram'] for item in row[architecture]))
        files['semantic_pairs.csv']=csv_text(['word_a','word_b','CBOW_cosine','Skip-Gram_cosine'],([r['word_a'],r['word_b'],r['cbow'],r['skipgram']] for r in pairs))
        files['rare_words.csv']=csv_text(['word','frequency','CBOW_neighbors','Skip-Gram_neighbors'],([r['word'],r['frequency'],', '.join(r['cbow']),', '.join(r['skipgram'])] for r in rare))
        files['word_embedding_report.md']='# Word Embedding Report\n\n'+'\n\n'.join('## '+r['question']+'\n\n'+r['answer'] for r in evidence)+'\n\n## Why might Skip-Gram be particularly useful for rare or domain-specific scientific terms?\n\nSkip-Gram creates prediction signals from a center word to multiple context words, so a rare center word can receive several contextual updates when it appears. Whether that helps here must be judged from the rare-word experiment above, not assumed from theory.'
    return download_zip(files,'word2vec_results')


@require_http_methods(['GET','POST'])
def word2vec_view(request):
    source=request.POST if request.method=='POST' else request.GET
    filters=topic_context(request)
    selection_started=time.monotonic()
    selection=embeddings.select_corpus(Document.objects.all(),source)
    selection_seconds=time.monotonic()-selection_started
    records=selection['records']
    params=embeddings.parameters(source)
    comparison_settings=embeddings.experiment_settings(source)
    experiment_params={**params,**comparison_settings}
    cache_name=embeddings.preset_cache_name(selection,params)
    language=request.COOKIES.get('biomedir_language','en')
    if request.method=='POST':
        if not selection['valid']:
            messages.error(request,'語料數量無效；至少選一篇且不可超過可用數量。' if language=='zh-Hant' else 'Invalid corpus counts. Select at least one document without exceeding availability.')
        else:
            try:
                if request.POST.get('action')=='experiment':
                    base_models=embeddings.load_both(records,params,cache_name=cache_name)
                    if not all(base_models.values()):
                        base_models=embeddings.train_both(records,params,cache_name=cache_name)
                    requested=request.POST.get('experiment','all')
                    kinds=list(embeddings.EXPERIMENTS) if requested=='all' else [requested]
                    for kind in kinds:
                        embeddings.train_experiment(records,experiment_params,kind,base_models=base_models)
                    messages.success(request,'三組參數比較已完成並寫入快取。' if language=='zh-Hant' else 'All parameter comparisons were completed and cached.')
                else:
                    before=embeddings.cache_status(records,params,cache_name)
                    trained=embeddings.train_both(records,params,cache_name=cache_name)
                    embeddings.build_global_artifacts(records,params,trained,cache_name=cache_name)
                    if before['ready']:
                        messages.success(request,'已載入快取模型。' if language=='zh-Hant' else 'Cached models loaded.')
                    else:
                        messages.success(request,'CBOW 與 Skip-Gram 已使用相同語料與設定完成訓練並寫入永久快取。' if language=='zh-Hant' else 'CBOW and Skip-Gram were trained with identical data and settings and saved to persistent cache.')
            except ValueError as exc:
                text=str(exc)
                if language=='zh-Hant':
                    if 'No sentences' in text: text='沒有至少包含兩個 token 的句子；請增加摘要數量。'
                    elif 'Fewer than two' in text: text='符合條件的詞彙少於兩個；請降低 min_count 或增加摘要數量。'
                messages.error(request,text)
        query=[(key,str(params[key])) for key in ['condition','dimensions','window','min_count','epochs']]
        query += [('corpus_preset',selection['preset']),('medical_count',selection['requested']['medical']),('cs_count',selection['requested']['cs'])]
        query += list(comparison_settings.items())
        for name,limit in [('targets',1000),('local_word',120),('experiment',40)]:
            if request.POST.get(name): query.append((name,request.POST[name][:limit]))
        anchor='#parameter-comparison' if request.POST.get('action')=='experiment' else '#training-setup'
        return redirect(reverse('search:word2vec')+'?'+urlencode(query)+anchor)

    cache_load_started=time.monotonic()
    models=embeddings.load_both(records,params,cache_name=cache_name)
    global_artifacts=embeddings.load_global_artifacts(records,params,cache_name=cache_name) if all(models.values()) else None
    cache_load_seconds=time.monotonic()-cache_load_started
    status=embeddings.cache_status(records,params,cache_name,cache_load_seconds)
    logger.info('Word2Vec corpus selection: %.4f sec; cache load: %.4f sec',selection_seconds,cache_load_seconds)
    stats=dict(global_artifacts['statistics']) if global_artifacts else embeddings.corpus_statistics(records,params['condition'])
    if global_artifacts:
        frequencies=None
    else:
        frequencies=stats.pop('frequencies')
        stats.pop('sentences_data')
    both_ready=all(models.values())
    targets=[];neighbor_rows=[];pair_rows=[];rare_rows=[];comparison_rows=[]
    global_visualizations={};local_neighbors={};global_word_list=[]
    local_word=request.GET.get('local_word','').strip()[:120];error=None;targets_text=request.GET.get('targets','')[:1000]
    if both_ready:
        targets=embeddings.parse_targets(targets_text,models,params['condition']);targets_text=', '.join(targets)
        neighbor_rows=embeddings.neighbor_comparison(models,targets)
        pair_rows=embeddings.semantic_pairs(models);rare_rows=global_artifacts['rare'] if global_artifacts else embeddings.rare_word_analysis(models,frequencies)
        comparison_rows=embeddings.comparison_table(models,rare_rows)
        if global_artifacts:
            global_word_list=global_artifacts['words']
            global_visualizations=global_artifacts['visualizations']
        if local_word:
            local_started=time.monotonic()
            try:
                local_word=embeddings.normalize_word(local_word,params['condition'])
                common=set(models['cbow'].wv.index_to_key)&set(models['skipgram'].wv.index_to_key)
                if local_word not in common: raise ValueError('The target word is outside the shared vocabulary.')
                highlighted=[]
                for name,model in models.items():
                    rows=[{'term':term,'similarity':float(score)} for term,score in model.wv.most_similar(local_word,topn=min(10,len(model.wv)-1))]
                    local_neighbors[name]=rows
                    highlighted.extend(row['term'] for row in rows)
                query_words=[word for word in dict.fromkeys([local_word,*global_word_list,*highlighted]) if word in common]
                searched_visualizations={}
                for name,model in models.items():
                    visual=embeddings.semantic_visualizations(model,query_words,selected=local_word)
                    neighbor_terms={row['term'] for row in local_neighbors[name]}
                    for chart in (visual['pca'],visual['tsne']):
                        for point in chart['points']:
                            point['highlight']='target' if point['term']==local_word else ('neighbor' if point['term'] in neighbor_terms else '')
                    searched_visualizations[name]=visual
                global_word_list=query_words
                global_visualizations=searched_visualizations
            except (ValueError,IndexError) as exc:
                error=str(exc)
                if language=='zh-Hant': error='搜尋詞不在兩個模型的共同詞彙中。'
            finally:
                logger.info('Word2Vec highlighted map query %r: %.4f sec',local_word,time.monotonic()-local_started)
    experiment_cards=[]
    for experiment_kind in embeddings.EXPERIMENTS:
        kind,label,configs=embeddings.experiment_config(experiment_params,experiment_kind)
        experiment_cards.append({
            'kind':kind,
            'label':label,
            'values':[config[kind] for config in configs],
            'limits':embeddings.EXPERIMENT_LIMITS[kind],
            'result':embeddings.load_experiment_result(records,experiment_params,kind),
        })
    loss_series=[{'label':'CBOW','points':models['cbow'].bir_metadata.get('losses',[])},{'label':'Skip-Gram','points':models['skipgram'].bir_metadata.get('losses',[])}] if both_ready else []
    training_sample=models['cbow'].bir_metadata.get('sample_preview') if both_ready else {}
    choices=[{'name':name,'label':label,'options':[{'value':value,'label':str(value),'selected':params[name]==value} for value in options]} for name,label,options in [('condition','Preprocessing',['B','C','D']),('dimensions','Embedding dimension',[50,100]),('window','Context window',[2,5,10]),('min_count','Minimum word count',[1,2,5]),('epochs','Training epochs',[10,20,40])]]
    return render(request,'search/word2vec.html',{**filters,'analysis_tab':'word2vec','page_title':'Word2Vec · BioMed IR',
        'selection':selection,'params':params,'full_params':embeddings.full_parameters(params),'parameter_choices':choices,
        'cache_status':status,'cache_name':cache_name,'selection_seconds':selection_seconds,
        'corpus_stats':stats,'both_ready':both_ready,'paired_models':models,'training_sample':training_sample,'loss_series':loss_series,
        'targets':targets,'targets_text':targets_text,'neighbor_comparison':neighbor_rows,'semantic_pairs':pair_rows,'rare_rows':rare_rows,
        'comparison_rows':comparison_rows,'global_words':global_word_list,'global_visualizations':global_visualizations,
        'local_word':local_word,'local_neighbors':local_neighbors,'embedding_error':error,
        'experiment_cards':experiment_cards,'comparison_settings':comparison_settings,
        'experiment_choices':[{'value':key,'label':value[0]} for key,value in embeddings.EXPERIMENTS.items()]})


def word2vec_export(request):
    selection=embeddings.select_corpus(Document.objects.all(),request.GET);records=selection['records']
    params=embeddings.parameters(request.GET);cache_name=embeddings.preset_cache_name(selection,params)
    models=embeddings.load_both(records,params,cache_name=cache_name)
    if not all(models.values()): raise Http404('Train both models for this exact corpus and parameter combination first.')
    stats=embeddings.corpus_statistics(records,params['condition']);targets=embeddings.parse_targets(request.GET.get('targets',''),models,params['condition'])
    comparisons=embeddings.neighbor_comparison(models,targets);pairs=embeddings.semantic_pairs(models);rare=embeddings.rare_word_analysis(models,stats['frequencies'])
    artifacts=embeddings.load_global_artifacts(records,params,cache_name=cache_name)
    if artifacts is None: raise Http404('Cached visualization artifacts are missing. Train this exact model pair again.')
    words=artifacts['words'];files={}
    for name,model in models.items():
        files[f'{name}_vectors.txt']=f'{len(model.wv)} {model.vector_size}\n'+'\n'.join(word+' '+' '.join(format(float(x),'.8g') for x in model.wv[word]) for word in model.wv.index_to_key)+'\n'
        files[f'{name}_metadata.json']=json.dumps(model.bir_metadata,ensure_ascii=False,indent=2)
        files[f'{name}_training_loss.csv']=csv_text(['epoch','loss','cumulative_loss','seconds'],([row[key] for key in ['epoch','loss','cumulative_loss','seconds']] for row in model.bir_metadata['losses']))
        visual=artifacts['visualizations'][name];files[f'{name}_pca.json']=json.dumps(visual['pca'],ensure_ascii=False,indent=2);files[f'{name}_tsne.json']=json.dumps(visual['tsne'],ensure_ascii=False,indent=2)
    files['manifest.json']=json.dumps(embeddings.load_manifest(records,params,cache_name),ensure_ascii=False,indent=2)
    files['corpus.json']=json.dumps({key:value for key,value in selection.items() if key!='records'},ensure_ascii=False,indent=2)
    files['corpus_ids.csv']=csv_text(['local_id','domain'],([record[0],record[2]] for record in records))
    files['model_comparison.csv']=csv_text(['property','CBOW','Skip-Gram'],embeddings.comparison_table(models,rare))
    files['semantic_neighbors.csv']=csv_text(['target','architecture','rank','neighbor','cosine_similarity'],([row['target'],architecture,item['rank'],item['term'],item['similarity']] for row in comparisons for architecture in ['cbow','skipgram'] for item in row[architecture]))
    files['semantic_pairs.csv']=csv_text(['word_a','word_b','CBOW_cosine','Skip-Gram_cosine'],([row['word_a'],row['word_b'],row['cbow'],row['skipgram']] for row in pairs))
    files['rare_words.csv']=csv_text(['word','frequency','architecture','rank','neighbor','cosine_similarity'],([row['word'],row['frequency'],architecture,rank,item['term'],item['similarity']] for row in rare for architecture in ['cbow','skipgram'] for rank,item in enumerate(row[architecture],1)))
    return download_zip(files,'word2vec_results')

def research_view(request):
    filters=topic_context(request);documents=filter_documents(Document.objects.all(),filters['selected_topics']);analyses=analyze_queryset(documents)
    condition=request.GET.get('condition','D')
    if condition not in CONDITIONS:condition='D'
    selected=analyses[condition];research=report(analyses,condition)
    selected_term=request.GET.get('report_term','')[:120]
    if not selected_term and selected['rows']:selected_term=selected['rows'][0]['term']
    idf=next((r['idf'] for r in selected['rows'] if r['term']==selected_term),None)
    tfidf=[]
    if idf is not None:
        for doc in documents.order_by('pk'):
            tf=condition_tokens(doc.abstract)[condition].count(selected_term)
            if tf:tfidf.append({'document':doc,'tf':tf,'idf':idf,'tfidf':tf*idf})
        tfidf.sort(key=lambda row:(-row['tfidf'],row['document'].pk))
    return render(request,'search/research.html',{**filters,**research,'analysis_tab':'research','page_title':'Research report · BioMed IR','selected':selected,'condition':condition,'analyses':list(analyses.values()),'cfdf_rows':selected['rows'][:20],'report_term':selected_term,'tfidf_page':Paginator(tfidf,20).get_page(request.GET.get('page',1)),'pmid_count':documents.exclude(pmid='').exclude(abstract='').count(),'arxiv_count':documents.exclude(arxiv_id='').exclude(abstract='').count(),'summary_only':request.GET.get('summary')=='1'})

def research_export(request):
    filters=topic_context(request);documents=filter_documents(Document.objects.all(),filters['selected_topics']);analyses=analyze_queryset(documents)
    condition=request.GET.get('condition','D')
    if condition not in CONDITIONS:condition='D'
    scope=' + '.join(t.name for t in filters['topics'] if t.selected) or 'All articles'
    research=report(analyses,condition)
    idfs={r['term']:r['idf'] for r in analyses[condition]['rows']};rows=[]
    for doc in documents.order_by('pk'):
        for term,tf in Counter(condition_tokens(doc.abstract)[condition]).items():
            rows.append([doc.pk,doc.pmid,doc.arxiv_id,term,tf,idfs[term],tf*idfs[term]])
    files={'research_report.md':markdown(analyses,condition,scope),'executive_summary.md':'# Executive summary\n\n'+'\n\n'.join(research['summary']),'tfidf.csv':csv_text(['local_id','PMID','arXiv_ID','term','TF','IDF_log10','TF_IDF'],rows),'CF_DF_IDF_top50.csv':csv_text(['term','CF','DF','IDF'],([r[k] for k in ['term','cf','df','idf']] for r in analyses[condition]['rows'][:50]))}
    return download_zip(files,'research_report')

def distance_view(request):
    filters=topic_context(request)
    query=request.GET.get('term','cancer').strip()[:32]
    normalized=normalize(query,'NFC',True)
    data=analyze_queryset(filter_documents(Document.objects.all(),filters['selected_topics']))['B']
    groups={distance:[] for distance in range(1,5)}
    if normalized:
        for row in data['rows']:
            candidate=normalize(row['term'],'NFC',True)
            if abs(len(normalized)-len(candidate))>4:
                continue
            distance=bounded_distance(normalized,candidate,4)
            if 1<=distance<=4:
                groups[distance].append({'term':row['term'],'distance':distance,'cf':row['cf'],'df':row['df']})
    candidate_groups=[]
    all_candidates=[]
    for distance,rows in groups.items():
        rows.sort(key=lambda row:(-row['df'],-row['cf'],row['term']))
        candidate_groups.append({'distance':distance,'rows':rows,'count':len(rows)})
        all_candidates.extend(rows)
    requested=request.GET.get('candidate','')
    selected=next((row for row in all_candidates if row['term']==requested),None)
    if selected is None and all_candidates:
        selected=all_candidates[0]
    matrix=edit_matrix(normalized,normalize(selected['term'],'NFC',True)) if selected else None
    return render(request,'search/distance.html',{
        **filters,'analysis_tab':'distance','page_title':'Text matching · BioMed IR',
        'query':query,'normalized_query':normalized,'candidate_groups':candidate_groups,
        'candidate_count':len(all_candidates),'selected_candidate':selected,'matrix':matrix,
    })
