import csv
import io
import json
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
def word2vec_view(request):
    filters=topic_context(SimpleNamespace(GET=request.POST) if request.method=='POST' else request)
    documents=filter_documents(Document.objects.all(),filters['selected_topics'])
    records=embeddings.corpus_records(documents)
    params=embeddings.parameters(request.POST if request.method=='POST' else request.GET)
    if request.method=='POST':
        try:
            embeddings.train(records,params)
            messages.success(request,'Word2Vec trained on your current abstracts. Explore similar words below.')
        except ValueError as exc:messages.error(request,str(exc))
        query=[(k,str(v)) for k,v in params.items()]+[('topic',t) for t in filters['selected_topics']]
        if request.POST.get('word'): query.append(('word',request.POST['word'][:120]))
        return redirect(reverse('search:word2vec')+'?'+urlencode(query))
    model=embeddings.load(records,params)
    word=request.GET.get('word','').strip()[:120]
    error=None;neighbors=[];projection={};overview_projection={};normalized=''
    if model is not None and word:
        try:
            normalized,neighbors=embeddings.neighbors(model,word,params['condition'])
            projection=embeddings.projection(model,[normalized,*[r['term'] for r in neighbors]])
        except ValueError as exc:error=str(exc)
    vocabulary=[]
    if model is not None:
        vocabulary=[{'term':w,'cf':model.wv.get_vecattr(w,'count')} for w in model.wv.index_to_key[:20]]
        overview_projection=embeddings.vocabulary_projection(model)
    return render(request,'search/word2vec.html',{**filters,'analysis_tab':'word2vec','page_title':'Word2Vec · BioMed IR','params':params,'parameter_choices':[{'name':name,'label':label,'options':[{'value':value,'label':{'skipgram':'Skip-gram','cbow':'CBOW','B':'B · keep stopwords','C':'C · no stopwords','D':'D · Porter stems'}.get(value,str(value)),'selected':params[name]==value} for value in options]} for name,label,options in [('architecture','Architecture',['skipgram','cbow']),('condition','Preprocessing',['B','C','D']),('dimensions','Vector dimensions',[50,100]),('window','Context window',[2,5,10]),('min_count','Minimum word count',[1,2,5]),('epochs','Training epochs',[10,20,40])]],'model_metadata':model.bir_metadata if model is not None else None,'document_count':len(records),'word':word,'normalized_word':normalized,'neighbors':neighbors,'embedding_error':error,'projection':projection,'overview_projection':overview_projection,'model_vocabulary':vocabulary})

def word2vec_export(request):
    filters=topic_context(request);records=embeddings.corpus_records(filter_documents(Document.objects.all(),filters['selected_topics']))
    params=embeddings.parameters(request.GET);model=embeddings.load(records,params)
    if model is None:raise Http404('Train this scope and parameter combination first.')
    header=f'{len(model.wv)} {model.vector_size}\n'
    vectors=header+'\n'.join(w+' '+' '.join(format(float(x),'.8g') for x in model.wv[w]) for w in model.wv.index_to_key)+'\n'
    files={'vectors.txt':vectors,'metadata.json':json.dumps(model.bir_metadata,ensure_ascii=False,indent=2),'corpus_ids.csv':csv_text(['local_id'],([pk] for pk,_ in records))}
    files['vocabulary_pca.json']=json.dumps(embeddings.vocabulary_projection(model),ensure_ascii=False,indent=2)
    word=request.GET.get('word','')[:120]
    if word:
        try:
            normalized,rows=embeddings.neighbors(model,word,params['condition'])
            files['similar_words.csv']=csv_text(['query','neighbor','cosine_similarity'],([normalized,r['term'],r['similarity']] for r in rows))
            files['projection.json']=json.dumps(embeddings.projection(model,[normalized,*[r['term'] for r in rows]]))
        except ValueError:pass
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
    files={'research_report.md':markdown(analyses,condition,scope),'executive_summary.md':'# Executive summary\n\n'+'\n\n'.join(research['summary']),'tfidf.csv':csv_text(['local_id','PMID','arXiv_ID','term','TF','IDF_ln','TF_IDF'],rows),'CF_DF_IDF_top50.csv':csv_text(['term','CF','DF','IDF'],([r[k] for k in ['term','cf','df','idf']] for r in analyses[condition]['rows'][:50]))}
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
