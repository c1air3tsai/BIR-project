"""Local arXiv abstract CSV fallback when the public API is rate limited."""
import csv
import hashlib
import io
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from django.conf import settings
from django.db import transaction
from .indexer import index_file, DuplicateDocumentError

def import_abstract_csv(upload,topic):
    if not upload.name.lower().endswith('.csv'):raise ValueError('Choose a UTF-8 CSV file.')
    if upload.size>20*1024*1024:raise ValueError('CSV limit is 20 MB.')
    try:text=upload.read().decode('utf-8-sig')
    except UnicodeDecodeError as exc:raise ValueError('Save the CSV as UTF-8 before uploading.') from exc
    reader=csv.DictReader(io.StringIO(text))
    if not {'id','title','abstract'}<=set(reader.fieldnames or []):raise ValueError('CSV headers must include id,title,abstract. id is the actual arXiv identifier.')
    rows=list(reader)
    if not 1<=len(rows)<=1000:raise ValueError('Upload 1–1,000 CSV rows at once.')
    result={'new':0,'reused':0,'errors':[]}
    corpus=Path(settings.BASE_DIR)/'data'/'corpus';corpus.mkdir(parents=True,exist_ok=True)
    for number,row in enumerate(rows,2):
        path=None
        try:
            identifier=re.sub(r'v\d+$','',(row.get('id') or '').strip().removeprefix('https://arxiv.org/abs/'))
            if not re.fullmatch(r'(?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})',identifier):raise ValueError('Invalid arXiv ID.')
            title=(row.get('title') or '').strip();abstract=(row.get('abstract') or '').strip()
            if not title or not abstract:raise ValueError('Title and abstract must be nonempty.')
            root=ET.Element('PubmedArticleSet');article=ET.SubElement(root,'PubmedArticle');body=ET.SubElement(ET.SubElement(article,'MedlineCitation'),'Article')
            ET.SubElement(body,'ArticleTitle').text=title;ET.SubElement(ET.SubElement(body,'Abstract'),'AbstractText').text=abstract
            journal=ET.SubElement(body,'Journal');ET.SubElement(journal,'Title').text='arXiv (CSV import)'
            ids=ET.SubElement(ET.SubElement(article,'PubmedData'),'ArticleIdList');ET.SubElement(ids,'ArticleId',IdType='arxiv').text=identifier
            digest=hashlib.sha256((identifier+title+abstract).encode()).hexdigest()[:16];path=corpus/('arXiv_csv_'+digest+'.xml')
            existed=path.exists()
            with transaction.atomic():
                if not existed:path.write_bytes(ET.tostring(root,encoding='utf-8',xml_declaration=True))
                try:doc=index_file(str(path));result['new']+=1
                except DuplicateDocumentError as exc:
                    doc=exc.document;result['reused']+=1
                    if not existed:path.unlink(missing_ok=True)
                if topic:doc.topics.add(topic)
        except (ValueError,ET.ParseError) as exc:
            if path and not existed:path.unlink(missing_ok=True)
            result['errors'].append(f'Row {number}: {exc}')
        except Exception:
            if path and not existed:path.unlink(missing_ok=True)
            raise
    return result
