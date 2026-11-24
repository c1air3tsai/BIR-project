# =============================================================================
# indexing/inverted_index.py
# Part 1：基本的 Inverted Index（後面 BSBI、SPIMI 的比較基準），與三種查詢
#
#   Dictionary   ：term → df（幾篇文件含有它）＋ 指向 postings list 的位置
#                  查詢時先查 dictionary，才知道有沒有這個 term、它的 postings 在哪裡
#   Postings list：含有這個 term 的所有 docID，由小到大排好
#                  排好序才能用「兩個指標一起往前走」做 AND / OR，不需要兩兩比對
#
#   InvertedIndex：全部放在記憶體（Part 1），一次讀完所有文件直接建好
#   DiskIndex    ：讀 BSBI / SPIMI 寫到磁碟的最終索引（dictionary 讀進記憶體，postings 查詢時才從檔案讀）
#
#   查詢：single-term（直接回傳 postings）、AND（intersect）、OR（union）
# =============================================================================
from array import array
from pathlib import Path

from .collection import preprocess, read_documents
from .storage import DOC_ID, read_dictionary


# AND：兩個排序好的 postings 各一個指標，docID 相同就收下，較小的那邊往前走
def intersect(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    return result


# OR：同樣兩個指標，每次收下較小的 docID（相同的只收一次）
def union(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# 查詢詞也做同樣的前處理（轉小寫等），才找得到 dictionary 裡的 term；被前處理刪掉（stopword、數字）就是空字串
def normalize(term):
    terms = preprocess(term or '')
    return terms[0] if terms else ''


class _Queries:
    def postings(self, term):
        raise NotImplementedError

    def and_query(self, a, b):
        return intersect(self.postings(a), self.postings(b))

    def or_query(self, a, b):
        return union(self.postings(a), self.postings(b))

    # 一組查詢（兩個 terms）的四個結果：兩個 single-term、AND、OR
    def boolean(self, a, b):
        left, right = self.postings(a), self.postings(b)
        return {'a': list(left), 'b': list(right), 'and': intersect(left, right), 'or': union(left, right)}


class InvertedIndex(_Queries):
    def __init__(self, documents=None, source='reuters'):
        self.index = {}                          # term → postings list（docID 由小到大）
        self.documents = self.tokens = 0
        for doc_id, text in (read_documents(source=source) if documents is None else documents):
            self.documents += 1
            for term in preprocess(text):
                self.tokens += 1
                postings = self.index.get(term)
                if postings is None:
                    self.index[term] = [doc_id]
                elif postings[-1] != doc_id:       # 同一篇文件只記一次
                    postings.append(doc_id)

    # Vocabulary：所有不重複的 terms（依字母排序）
    def vocabulary(self):
        return sorted(self.index)

    def df(self, term):
        return len(self.index.get(term, ()))

    def postings(self, term):
        return self.index.get(term, [])

    def pairs(self):
        return sum(len(postings) for postings in self.index.values())


class DiskIndex(_Queries):
    def __init__(self, folder):
        self.folder = Path(folder)
        self.dictionary = read_dictionary(self.folder)

    def vocabulary(self):
        return sorted(self.dictionary)

    def df(self, term):
        return self.dictionary.get(term, (0, 0))[0]

    def postings(self, term):
        if term not in self.dictionary:
            return array(DOC_ID)
        df, offset = self.dictionary[term]
        postings = array(DOC_ID)
        with open(self.folder / 'postings.bin', 'rb') as stream:
            stream.seek(4 * offset)
            postings.frombytes(stream.read(4 * df))
        return postings
