# -*- encoding: utf-8 -*-

# conda install torchtext
# pip install sentencepiece

#%matplotlib inline
import torch
import torchtext.vocab as vocab
import sys
sys.path.append('../..')
import dl_common_pytorch as dl
from torch import nn
import torch.utils.data as Data
print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

print([key for key in vocab.pretrained_aliases.keys() if "glove" in key])
cache_dir = "./GloVe"
glove = vocab.GloVe(name='6B', dim=50, cache=cache_dir)
print("一共包含%d个词。" % len(glove.stoi))
print(glove.stoi['beautiful'], glove.itos[3366])

# 求近义词，由于词向量空间中的余弦相似性可以衡量词语含义的相似性（为什么？），我们可以通过寻找空间中的k近邻，来查询单词的近义词。
def knn(W, x, k):
    '''
    @params:
        W: 所有向量的集合
        x: 给定向量
        k: 查询的数量
    @outputs:
        topk: 余弦相似性最大k个的下标
        [...]: 余弦相似度
    '''
    cos = torch.matmul(W, x.view((-1,))) / ((torch.sum(W * W, dim=1) + 1e-9).sqrt() * torch.sum(x * x).sqrt())
    _, topk = torch.topk(cos, k=k)
    topk = topk.cpu().numpy()
    return topk, [cos[i].item() for i in topk]

def get_similar_tokens(query_token, k, embed):
    '''
    @params:
        query_token: 给定的单词
        k: 所需近义词的个数
        embed: 预训练词向量
    '''
    topk, cos = knn(embed.vectors, embed.vectors[embed.stoi[query_token]], k+1)
    for i, c in zip(topk[1:], cos[1:]):  # 除去输入词
        print('cosine sim=%.3f: %s' % (c, (embed.itos[i])))

get_similar_tokens('chip', 3, glove)
'''
cosine sim=0.856: chips
cosine sim=0.749: intel
cosine sim=0.749: electronics
'''
get_similar_tokens('baby', 3, glove)
'''
cosine sim=0.839: babies
cosine sim=0.800: boy
cosine sim=0.792: girl
'''
get_similar_tokens('beautiful', 3, glove)
'''
cosine sim=0.921: lovely
cosine sim=0.893: gorgeous
cosine sim=0.830: wonderful
'''
# 除了求近义词以外，我们还可以使用预训练词向量求词与词之间的类比关系，例如“man”之于“woman”相当于“son”之于“daughter”。
# 求类比词问题可以定义为：对于类比关系中的4个词“a之于b相当于c之于d”，给定前3个词a/b/c求d。
# 类比词的思路是，搜索与vec(c)+vec(b)-vec(a)的结果向量最相似的词向量，其中vec(w)为w的词向量。
def get_analogy(token_a, token_b, token_c, embed):
    '''
    @params:
        token_a: 词a
        token_b: 词b
        token_c: 词c
        embed: 预训练词向量
    @outputs:
        res: 类比词d
    '''
    vecs = [embed.vectors[embed.stoi[t]] for t in [token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.vectors, x, 1)
    res = embed.itos[topk[0]]
    print(res)
    return res

get_analogy('man', 'woman', 'son', glove) # daughter

get_analogy('beijing', 'china', 'tokyo', glove) # japan

get_analogy('bad', 'worst', 'big', glove) # biggest

get_analogy('do', 'did', 'go', glove) # went