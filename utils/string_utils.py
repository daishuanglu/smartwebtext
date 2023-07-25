from itertools import cycle
from fastDamerauLevenshtein import damerauLevenshtein
import numpy as np
from numpy.linalg import norm
from numpy import dot
from rake_nltk import Rake
import fasttext
import os
import re
import torch
from torch.autograd import Function
import scipy.linalg
import string
from nltk import WordNetLemmatizer
from multiprocessing import Pool
from nltk.stem.snowball import SnowballStemmer

import yaml

def read_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

stemmer = SnowballStemmer("english")
wnl = WordNetLemmatizer()
rake=Rake()

COMPANY_NAME_SUFFICES = [
    'inc', 'llc', 'ltd', 'limited', 'corp', 'corporation', 'l.p.'
    'llp', 'lp', 'incorporated', 'plc', 'lc', 'l.c', 'co', 'n.a.', 's.a.']

def remove_company_suffix(s):
    for suffix in COMPANY_NAME_SUFFICES:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            return s.strip()
        if s.endswith(suffix+'.'):
            s = s[:-len(suffix)-1]
            return s.strip()
    return s

def lemmed_fast(text, lem_fn, cores=6):  # tweak cores as needed
    with Pool(processes=cores) as pool:
        result = pool.map(lem_fn, text)
    return result


def lemmed(text, lem_fn= None):
    if lem_fn is None:
        lem_fn = wnl.lemmatize
    return ' '.join([lem_fn(w) for w in text.split()])

punct_exclist = string.punctuation + string.digits


def remove_punct(s):
    punct_trans_table_ = str.maketrans(punct_exclist, ' ' * len(punct_exclist))
    s = s.lower().translate(punct_trans_table_)
    return s

def stopword_pattern(stopwords):
    return re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')


def load_stopwords(enc='iso-8859-1'):
    with open('utils/gensim_nltk_spacy_stopwords.txt', 'r', encoding=enc) as f:
        stopwords = [w[:-1] for w in f.readlines()]
    return stopwords


def remove_stopwords(text, stopwords, sub=''):
    stopword_ptn = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
    text = stopword_ptn.sub(sub, text)
    return text


class fasttext_toolkits():
    def __init__(self, fasttext_model_home="./fasttext"):
        self.fasttext_model_home=fasttext_model_home
        self.ft = fasttext.load_model(os.path.join(fasttext_model_home, 'cc.en.100.bin'))
        self.ft_lang = fasttext.load_model(os.path.join(fasttext_model_home, 'lid.176.bin'))
    # ([['__label__en']], [array([0.9331119], dtype=float32)]
    def detEN(self, sentences):
        return [i[0].split('__label__')[1] == 'en' for i in self.ft_lang.predict(sentences)[0]]

acct_info=['tic', 'sic', 'ni']

def split_by_multi_sep(s, seps):
    for sep in seps:
        s = s.replace(sep, '\t\t\t\t')
    return s.split('\t\t\t\t')

def ignore_unicode_char(s):
    string_encode = s.encode("ascii", "ignore")
    string_decode = string_encode.decode('utf-8')
    return string_decode

def clean_char(s):
    s = ignore_unicode_char(str(s))
    return s.replace('  ', ' ')


def replace_consecutive_spaces(text):
    # Use a regular expression to find consecutive spaces and replace them with a single space
    return re.sub(r'\s+', ' ', text)

def get_title_name(s, sep='-'):
    try:
        return ' '.join(s.lower().split(sep))
    except:
        return

def getcompanyname(company, sep='-'):
    try:
        company=re.sub("%\d\w", '', company.lower())
        company = re.sub("%\w\d", '', company)
        return ' '.join(company.split(sep))
    except:
        return

def find_links(comp_List, company_list_path):
    f = open(company_list_path, 'r', encoding='utf-8')
    #f = open('data_model/default_search_full.txt', 'r', encoding='utf-8')
    all_links = [l.rstrip() for l in f.readlines()]
    f.close()
    comp_names_from_link=[getcompanyname(c.lower().split('/')[-2]) for c in all_links]
    find_link=[]
    for c in comp_List:
        id=comp_names_from_link.index(c)
        if id>=0:
            find_link.append( all_links[id])
        else:
            print(c,' links not found.')
            find_link.append(str('None'))
    return find_link


def norm01(x):
    x= (x-x.min())/(x.max()-x.min())
    return x


def vec2mat_cos_sim(vec,mat):
    vec=np.atleast_2d(vec)
    mat=mat.transpose()
    p1 = vec.dot(mat)
    mat_norm=np.sqrt(np.einsum('ij,ij->j',mat,mat))
    mat_norm[mat_norm==0]=1e-3
    vec_norm=norm(vec)
    vec_norm=1e-3 if vec_norm==0 else vec_norm
    out1 = p1 / (mat_norm*vec_norm)
    #print(np.abs(np.linalg.norm(mat,axis=0)-mat_norm).max())
    return out1


def remove_duplicates(list_of_phrases):
    phrases=set(list_of_phrases)
    res=[]
    for p in phrases:
        if p=='nan': continue
        if len(p)<3: continue
        if len(res)==0:
            res.append(p)
        else:
            sim=[edit(w.split(), p.split())>=0.5 for w in res]
            if any(sim):
                continue
            res.append(p)
    return res


def correct_singles(d):
    d['analysis']+= d.pop('analysi')
    return d


def remove_all_zeros_col(OriginMat):
    id=~np.all(OriginMat == 0, axis = 0)
    return id,OriginMat[:,id]


class ListNode():
    def __init__(self, value,next=None):
        self.val = value
        self.next=next


def charlinkedlist(s):
    words = list(s)
    head = None
    p = head
    for w in words:
        if head == None:
            head = ListNode(w)
            p = head
        else:
            p.next = ListNode(w)
            p = p.next
    return head


def str2linkedlist(s):
    words=s.split(' ')
    head = None
    p=head
    for w in words:
        if head==None:
            head=ListNode(w)
            p=head
        else:
            p.next=ListNode(w)
            p=p.next
    return head


def linkedlist2str(head):
    s=''
    while head:
        s += head.val+' '
        head=head.next
    return s[:-1]


def llmatch(head, words,th=0.5):
    nw = len(words)
    h = ListNode('idle')
    h.next = head
    p = h
    q = p.next
    nfound = 0
    ws = cycle(words)
    w = next(ws)
    while q:
        if q.val == w:
            nfound += 1
            w = next(ws)
        else:
            p = q
        q = q.next
        if nfound / nw >= th:
            for _ in range(nw - nfound):
                if q:
                    q = q.next
            p.next = q
            if not q:
                break
            q = q.next
            nfound = 0

    return h.next


def nchar_diff(word1, word2):
    if word1==word2:
        return 0
    if len(word1)>len(word2):
        w,t = word2,word1
    else:
        w,t = word1,word2
    ndiff = len(t)- len(w)
    i=0
    for i in range(len(w)):
        if w[i]==t[i]:
            continue
        else:
            break
    return ndiff+len(w)-i

def char_match(s1,target,ratio=True):
    if ratio:
        head = charlinkedlist(s1)
        head = llmatch(head, list(target), th=1.0)
        s1_recon = linkedlist2str(head).replace(' ', '')
        r=len(s1_recon)/len(target)
        return 1-r
    else:
        return nchar_diff(s1,target)


def containkwOR(sentence, keywords):
    for word in keywords:
        if word in sentence:
            return word
    return ''


def containkwAND(sentence, keywords):
    return all([w in sentence for w in keywords])



def clean_web_str_regex(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def findLocalWords(str1, keywords_rule,center='analytics', nlocal=12):
    words_list = clean_web_str_regex(str1.lower()).split(center)
    if containkwOR(' '.join(words_list[0].split()[-(nlocal + 1):]), keywords_rule):
        return True
    if containkwOR(' '.join(words_list[-1].split()[:nlocal]), keywords_rule):
        return True
    for words in words_list[1:-1]:
        if containkwOR(' '.join(words.split()[:nlocal]), keywords_rule):
            return True
        if containkwOR(' '.join(words.split()[-(nlocal + 1):]), keywords_rule):
            return True
    return False


def edit(s1,s2):
    return damerauLevenshtein(s1,s2,similarity=True)


def find_stock_code(s):
    s=s.replace(' ','')
    stocks=[]
    while True:
        r=re.search('Nasdaq:(.+?)\)',s )
        if r:
            stocks.append(r.group(1))
            s=s.replace(r.group(),'')
        else:
            break
    while True:
        r=re.search('NASDAQ:(.+?)\)',s )
        if r:
            stocks.append(r.group(1))
            s=s.replace(r.group(),'')
        else:
            break
    while True:
        r=re.search('NYSE:(.+?)\)',s )
        if r:
            stocks.append(r.group(1))
            s=s.replace(r.group(),'')
        else:
            break
    return stocks

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def keep_alpha(s):
    return re.sub(r"[^A-Za-z.']+", ' ', s)


def keep_alphanum(s):
    return re.sub(r"[^A-Za-z0-9.']+", ' ', s)

def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))


def split_to_sentences(string):
    return re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", string)


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply

def unionfind_common_words(phrases):
    phrase_sub=['' for _ in phrases]
    phrases_set=[set(p.split()) for p in phrases]
    omega=set(range(len(phrases)))
    found=set()
    while len(found)<len(omega):
        i=min(omega-found)
        p=phrases_set[i]
        found.add(i)
        found_i={i}
        repr=''
        lrepr=0
        for j in omega-found:
            common=list(p & phrases_set[j])
            if len(common)>0:
                found.add(j)
                found_i.add(j)
            if len(common)>lrepr:
                repr=' '.join( common)
                lrepr=len(common)
        if not repr:
            repr=phrases[i]
        for j in found_i:
            phrase_sub[j]=repr
        print(len(found), len(omega))
    return phrase_sub


def unionfind_common_string(words,th=3):
    w_set=list(set(words))
    s_sub = {w: '' for w in w_set}
    omega=set(range(len(w_set)))
    found=set()
    reduced_size = 0
    while len(found)<len(omega):
        i=min(omega-found)
        w=words[i]
        s_sub[w] = w
        found.add(i)
        for j in omega-found:
            iscommon= char_match(w,w_set[j], ratio=False)<=th
            if iscommon:
                found.add(j)
                s_sub[w_set[j]]=w
        print(len(found), len(omega))
        reduced_size+=1
    print('reduced size to ', reduced_size)
    return s_sub