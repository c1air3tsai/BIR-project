"""Distance-1/2 spelling alternatives from the filtered abstract vocabulary."""
import hashlib
import unicodedata
from collections import Counter, defaultdict
from django.core.cache import cache
from .text_processing import STOPWORDS, tokenize, iter_token_matches


def canonical(word):
    return unicodedata.normalize('NFC', word.casefold())


def bounded_distance(left, right, limit):
    """Levenshtein in a narrow band; return limit+1 when too distant."""
    if abs(len(left)-len(right)) > limit:
        return limit+1
    previous=list(range(len(right)+1))
    for i,a in enumerate(left,1):
        current=[limit+1]*(len(right)+1)
        current[0]=i
        start,end=max(1,i-limit),min(len(right),i+limit)
        for j in range(start,end+1):
            current[j]=min(current[j-1]+1,previous[j]+1,previous[j-1]+(a!=right[j-1]))
        if min(current[max(0,start-1):end+1]) > limit:
            return limit+1
        previous=current
    return previous[-1]


def vocabulary(documents):
    records=list(documents.order_by('pk').values_list('pk','abstract'))
    digest=hashlib.sha256()
    for pk,text in records:
        digest.update(f'{pk}\0{text}\0'.encode('utf-8'))
    key='spelling-scope-v2-identifiers-'+digest.hexdigest()
    result=cache.get(key)
    if result is not None:
        return result
    cf,df=Counter(),Counter()
    for _,text in records:
        words=[canonical(w) for w in tokenize(unicodedata.normalize('NFC',text))]
        cf.update(words);df.update(set(words))
    lengths=defaultdict(list)
    for word in cf:
        # Include biomedical identifiers such as GLP-1/GLP1. Pure numbers and
        # stopwords are still excluded from spelling alternatives.
        if 2<=len(word)<=32 and any(char.isalpha() for char in word) and word not in STOPWORDS:
            lengths[len(word)].append(word)
    result={'cf':cf,'df':df,'lengths':dict(lengths)}
    cache.set(key,result,300)
    return result


def suggest_query(query, documents):
    if not query or len(query)>512:
        return []
    matches=list(iter_token_matches(query))
    if len(matches)>12:
        return []
    data=vocabulary(documents)
    if not data['cf']:
        return []
    alternatives=[]
    for match in matches:
        original=query[match.start():match.end()];word=canonical(original)
        if (not 2<=len(word)<=32 or not any(char.isalpha() for char in word)
                or word in STOPWORDS):
            continue
        limit=2
        options=[]
        for size in range(len(word)-limit,len(word)+limit+1):
            for candidate in data['lengths'].get(size,[]):
                if candidate==word:
                    continue
                distance=bounded_distance(word,candidate,limit)
                if 1<=distance<=limit:
                    options.append((distance,-data['df'][candidate],-data['cf'][candidate],candidate))
        options.sort()
        alternatives.extend((match,option) for option in options)
    results=[];seen=set()
    for match,(distance,negative_df,negative_cf,word) in sorted(
            alternatives,key=lambda item:(item[1][0],item[1][1],item[1][2],item[1][3],item[0].start())):
        original=query[match.start():match.end()]
        if original.isupper(): word=word.upper()
        elif original.istitle(): word=word.title()
        corrected=query[:match.start()]+word+query[match.end():]
        if corrected in seen or corrected==query:
            continue
        seen.add(corrected)
        results.append({'query':corrected,'distance':distance,'corrected_words':1})
    return results
