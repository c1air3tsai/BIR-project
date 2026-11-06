# =============================================================================
# reuters/toy_word2vec.py
# Project 4 學習目標 3：自己用 numpy 實作 CBOW 與 Skip-Gram 的基本訓練流程
#
# 真正的實驗用 gensim（reuters/embedding.py），速度快、有 negative sampling 等最佳化；
# 這個檔案是「看得懂每一步」的教學版：小詞彙、完整 softmax、逐步更新。
#
# 兩個權重矩陣：
#   W_in  (V × D)：輸入詞向量，訓練完就是我們要的「詞向量」
#   W_out (D × V)：輸出層權重
#
# CBOW（上下文 → 中心詞）：
#   h = mean(W_in[上下文的詞])                 隱藏層 = 上下文詞向量的平均
#   p = softmax(h · W_out)                     預測每個詞是中心詞的機率
#   loss = −log p[中心詞]
#
# Skip-Gram（中心詞 → 上下文）：
#   h = W_in[中心詞]                           隱藏層 = 中心詞的向量
#   對每一個上下文詞 c：p = softmax(h · W_out)，loss += −log p[c]
#
# 更新（梯度下降）：e = p − onehot(目標)，
#   W_out −= lr · outer(h, e)；  W_in[輸入詞] −= lr · (W_out · e)（CBOW 平分給每個上下文詞）
# =============================================================================
import numpy as np

# 示範語料：模仿 Reuters 五個主題的短句（讓詞彙很小，結果容易解讀）
TOY_SENTENCES = [
    'opec raised crude oil output', 'opec cut crude oil prices', 'saudi arabia raised oil output',
    'oil prices rose after opec meeting', 'wheat exports to china rose', 'corn exports to china rose',
    'farmers harvest wheat and corn', 'soviet union bought wheat and corn',
    'company reported net profit', 'company reported net loss', 'firm reported quarterly profit',
    'firm reported quarterly loss', 'firm agreed to acquire rival', 'company agreed merger with rival',
    'japan trade surplus widened', 'trade deficit with japan widened',
]


def build_vocabulary(sentences):
    words = sorted({word for sentence in sentences for word in sentence})
    return {word: i for i, word in enumerate(words)}


# 把句子轉成 (上下文詞編號清單, 中心詞編號) 的訓練樣本
def training_pairs(sentences, vocabulary, window=2):
    pairs = []
    for sentence in sentences:
        ids = [vocabulary[w] for w in sentence]
        for i, center in enumerate(ids):
            context = ids[max(0, i - window):i] + ids[i + 1:i + 1 + window]
            if context:
                pairs.append((context, center))
    return pairs


def softmax(x):
    exp = np.exp(x - x.max())
    return exp / exp.sum()


class ToyWord2Vec:
    def __init__(self, vocabulary, dim=10, seed=42):
        rng = np.random.default_rng(seed)
        self.vocabulary = vocabulary
        self.words = sorted(vocabulary, key=vocabulary.get)
        size = len(vocabulary)
        self.W_in = rng.normal(0, 0.1, (size, dim))
        self.W_out = rng.normal(0, 0.1, (dim, size))

    # CBOW 一步：回傳 loss 與這一步的中間值（網頁逐步說明用）
    def cbow_step(self, context, center, lr):
        h = self.W_in[context].mean(axis=0)              # 隱藏層 = 上下文向量平均
        p = softmax(h @ self.W_out)                       # 每個詞是中心詞的機率
        loss = -np.log(p[center])
        e = p.copy()
        e[center] -= 1.0                                  # e = p − onehot(中心詞)
        grad_h = self.W_out @ e
        self.W_out -= lr * np.outer(h, e)
        for word in context:                              # 梯度平分給每個上下文詞
            self.W_in[word] -= lr * grad_h / len(context)
        return loss, {'h': h, 'p_target': float(p[center])}

    # Skip-Gram 一步：中心詞要分別預測每一個上下文詞；
    # 每個 (中心詞, 上下文詞) 配對各自是一個訓練樣本，各更新一次
    def skipgram_step(self, context, center, lr):
        losses, probabilities = [], []
        for word in context:
            h = self.W_in[center].copy()                 # 隱藏層 = 中心詞向量
            p = softmax(h @ self.W_out)
            losses.append(-np.log(p[word]))
            probabilities.append(p[word])
            e = p.copy()
            e[word] -= 1.0                               # e = p − onehot(上下文詞)
            grad_h = self.W_out @ e
            self.W_out -= lr * np.outer(h, e)
            self.W_in[center] -= lr * grad_h
        return float(np.mean(losses)), {'p_target': float(np.mean(probabilities))}

    def train(self, pairs, architecture='cbow', epochs=200, lr=0.1, seed=0):
        rng = np.random.default_rng(seed)
        step = self.cbow_step if architecture == 'cbow' else self.skipgram_step
        losses = []
        for _ in range(epochs):
            total = 0.0
            for i in rng.permutation(len(pairs)):
                loss, _ = step(*pairs[i], lr)
                total += loss
            losses.append(total / len(pairs))
        return losses

    # 用 cosine 找最接近的詞
    def neighbors(self, word, topn=3):
        vectors = self.W_in / np.linalg.norm(self.W_in, axis=1, keepdims=True)
        sims = vectors @ vectors[self.vocabulary[word]]
        order = [i for i in np.argsort(-sims) if i != self.vocabulary[word]][:topn]
        return [(self.words[i], float(sims[i])) for i in order]


# 網頁用：兩種架構各訓練一次，回傳 loss 曲線、鄰近詞與第一個訓練樣本的說明
def demo(epochs=200, dim=10, window=2, lr=0.1):
    sentences = [s.split() for s in TOY_SENTENCES]
    vocabulary = build_vocabulary(sentences)
    pairs = training_pairs(sentences, vocabulary, window)
    words = sorted(vocabulary, key=vocabulary.get)
    result = {'vocabulary': len(vocabulary), 'pairs': len(pairs), 'sentences': TOY_SENTENCES,
              'example': {'context': [words[i] for i in pairs[2][0]], 'center': words[pairs[2][1]]}}
    for architecture in ('cbow', 'skipgram'):
        model = ToyWord2Vec(vocabulary, dim=dim)
        losses = model.train(pairs, architecture, epochs=epochs, lr=lr)
        result[architecture] = {
            'losses': losses,
            'neighbors': {word: model.neighbors(word) for word in ('oil', 'wheat', 'profit', 'japan')},
        }
    return result
