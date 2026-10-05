"""Independent topic groups; explicit overlap exclusion and deterministic balancing."""
from .analysis import analyze_queryset, chart_data
from .topic_filters import filter_documents
from .models import Document, Topic

def compare_domains(values):
    topics=list(Topic.objects.all())
    valid={str(t.pk) for t in topics}
    a=[x for x in values.getlist('domain_a') if x in valid]
    b=[x for x in values.getlist('domain_b') if x in valid]
    if not a and not b and 'domain_a' not in values and 'domain_b' not in values and not any(k in values for k in ['condition','sample','overlap']):
        a=[str(topics[0].pk)] if topics else []
        b=[str(topics[1].pk)] if len(topics)>1 else []
    groups=[]; sets=[]
    for ids in [a,b]:
        qs=filter_documents(Document.objects.exclude(abstract=''),ids) if ids else Document.objects.none()
        sets.append(set(qs.values_list('pk',flat=True)))
    overlap=sets[0]&sets[1]
    exclude=values.get('overlap','exclude')!='include'
    eligible=[ids-overlap if exclude else ids for ids in sets]
    cap=values.get('sample','500')
    cap=cap if cap in {'100','500','all'} else '500'
    size=min(len(ids) for ids in eligible) if eligible and all(eligible) else 0
    if cap!='all': size=min(size,int(cap))
    keys=['A','B']; condition=values.get('condition','D')
    if condition not in 'ABCD' or len(condition)!=1:condition='D'
    for i,ids in enumerate([a,b]):
        chosen=sorted(eligible[i]) if cap=='all' else sorted(eligible[i])[:size]
        docs=Document.objects.filter(pk__in=chosen)
        analyses=analyze_queryset(docs)
        data=analyses[condition]
        groups.append({'key':keys[i],'name':' + '.join(t.name for t in topics if str(t.pk) in ids) or 'Select topics','selected_ids':ids,'eligible':len(eligible[i]),'data':data,'analyses':analyses,'document_ids':chosen,'top10':data['rows'][:10]})
    chart={}
    for g in groups:
        d=chart_data({g['key']:g['data']})[g['key']]
        chart[g['key']]={**d,'key':g['key'],'label':g['name']}
    topa={r['term'] for r in groups[0]['top10']};topb={r['term'] for r in groups[1]['top10']}
    union=topa|topb
    return {'domain_topics':topics,'domain_groups':groups,'domain_a':a,'domain_b':b,'overlap_count':len(overlap),'overlap_mode':'exclude' if exclude else 'include','sample':cap,'domain_condition':condition,'domain_chart':chart,'top10_jaccard':len(topa&topb)/len(union) if union else None,'comparison_ready':all(g['data']['vocabulary']>1 for g in groups)}
