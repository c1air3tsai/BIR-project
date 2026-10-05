"""Actual local Word2Vec training; model keys include corpus content and settings."""
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from django.conf import settings
from .analysis import condition_tokens
from .text_processing import split_sentences

DEFAULTS = {'architecture':'skipgram','condition':'C','dimensions':100,'window':5,'min_count':2,'epochs':20}

def stable_hash(value):
    return int.from_bytes(hashlib.sha256(value.encode('utf-8')).digest()[:8], 'little')

def parameters(values):
    out = dict(DEFAULTS)
    for name, choices in {'architecture':['skipgram','cbow'], 'condition':['B','C','D'], 'dimensions':[50,100], 'window':[2,5,10], 'min_count':[1,2,5], 'epochs':[10,20,40]}.items():
        value=values.get(name, out[name])
        if isinstance(out[name], int):
            try: value=int(value)
            except (ValueError,TypeError): value=out[name]
        out[name]=value if value in choices else out[name]
    return out

def corpus_records(documents):
    return [(pk,text) for pk,text in documents.order_by('pk').values_list('pk','abstract') if text.strip()]

def model_path(records, params):
    digest=hashlib.sha256(json.dumps({'version':1,'params':params,'records':records},ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    return Path(settings.BASE_DIR)/'data'/'embeddings'/(digest+'.model')

def sentences_for(records, condition):
    return [tokens for _,text in records for sentence in split_sentences(text) if len(tokens:=condition_tokens(sentence)[condition])>=2]

def train(records, params):
    from gensim.models import Word2Vec
    sentences=sentences_for(records,params['condition'])
    if not sentences: raise ValueError('No sentences with at least two tokens. Import more abstracts first.')
    started=time.monotonic()
    model=Word2Vec(vector_size=params['dimensions'],window=params['window'],min_count=params['min_count'],sg=int(params['architecture']=='skipgram'),workers=1,seed=42,hashfxn=stable_hash,negative=5,hs=0,sample=0.001,epochs=params['epochs'])
    model.build_vocab(sentences)
    if len(model.wv)<2: raise ValueError('Fewer than two eligible words. Lower minimum count or import more abstracts.')
    model.train(sentences,total_examples=model.corpus_count,epochs=params['epochs'])
    metadata={'parameters':params,'documents':len(records),'sentences':len(sentences),'input_tokens':sum(map(len,sentences)),'training_vocabulary':len(model.wv),'seconds':round(time.monotonic()-started,3),'seed':42,'workers':1,'negative':5,'sample':0.001,'implementation':'gensim Word2Vec','scope':'Extracted abstract + Keywords only; sentence boundaries preserved'}
    model.bir_metadata=metadata
    destination=model_path(records,params);destination.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=destination.parent,suffix='.model');os.close(fd)
    try:
        model.save(temp,separately=[])
        os.replace(temp,destination)
    finally: Path(temp).unlink(missing_ok=True)
    return model

def load(records,params):
    path=model_path(records,params)
    if not path.exists(): return None
    from gensim.models import Word2Vec
    return Word2Vec.load(str(path))

def neighbors(model, text, condition, count=10):
    tokens=condition_tokens(text)[condition]
    if len(tokens)!=1: raise ValueError('Enter one word or biomedical token, such as learning or GLP-1.')
    word=tokens[0]
    if word not in model.wv: raise ValueError(f'“{word}” is outside this model vocabulary. Try a frequent term or lower minimum count.')
    rows=[{'term':term,'similarity':float(score)} for term,score in model.wv.most_similar(word,topn=min(count,len(model.wv)-1))]
    return word,rows

def projection(model, words):
    import numpy as np
    matrix=np.array([model.wv[w] for w in words],dtype=float)
    matrix-=matrix.mean(axis=0)
    _,singular,axes=np.linalg.svd(matrix,full_matrices=False)
    points=matrix@axes[:2].T
    total=float((singular**2).sum())
    return {'points':[{'term':word,'x':float(point[0]),'y':float(point[1]) if len(point)>1 else 0.0} for word,point in zip(words,points)],'variance':float((singular[:2]**2).sum()/total) if total else 0.0}


def vocabulary_projection(model, count=40):
    """PCA overview of the model's most frequent words."""
    words=model.wv.index_to_key[:min(count,len(model.wv))]
    result=projection(model,words)
    for rank,point in enumerate(result['points'],1):
        point['rank']=rank
        point['count']=int(model.wv.get_vecattr(point['term'],'count'))
    return result
