import numpy as np

from string_utils import wnl, stemmer
import time
from sklearn.metrics.pairwise import euclidean_distances

from collections import OrderedDict
from cf_imp import impute, imputor
from trm import predict_with_trm
from embedding_model import *
from ticker_utils import ticker_finder


def read_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def _lem_stem(kws):
    s = [wnl.lemmatize(w) for w in kws]
    s = [stemmer.stem(w) for w in s]
    return s


def _norm_emb_probs(emb):
    emb[emb < 0] = 0
    emb[emb > 1] = 1
    return emb


def local_global_modeling(model, df, _config):
    comp_all = sorted(set(df['Company'].tolist()))
    start_time=time.time()
    #words_list= [[singularize(w) for w in sent.split()] for sent in df['Text']]
    words_list = df['Text'].apply(lambda x: x.split()).tolist()
    print(int(time.time()-start_time),'secs. done.')
    model.fit_global(words_list)
    uni_vocab = set()
    comp_topics=OrderedDict()
    global_topics={
        'emb':model.global_embedding_matrix(),
        'vocab':model.global_vocab(),
        'tic':''
    }
    save(global_topics, _config['local']['embedding_path'])
    start_time = time.time()
    if 'local' not in _config.keys():
        return {}, global_topics
    for ii,comp in enumerate(comp_all):
        print('- company:', comp,'{}/{}'.format(ii,len(comp_all)))
        comp_words_list=np.array(words_list)[df['Company']==comp]
        fit=model.fit_local(comp_words_list, comp)
        if not fit:
            continue
        local_vocab = model.local_vocab(comp)
        ticker = ticker_finder(comp)
        comp_topics[comp]={
            'emb': model.local_embedding_matrix(comp),
            'vocab': local_vocab,
            'tic':ticker}
        uni_vocab.update( set(local_vocab.keys()) )
        print(int(time.time()-start_time),'secs')
    uni_vocab = {v:i for i,v in enumerate(sorted(uni_vocab))}
    save(global_topics, _config['global']['embedding_path'])
    if 'local' in _config.keys():
        save(comp_topics, _config['local']['embedding_path'])
    save(uni_vocab, _config['TRM']['uni_vocab'])
    return comp_topics, global_topics, uni_vocab


def construct_universal_emb(_config, comp_topics, uni_vocab):
    #comp_topics=self.comp_topics.copy()
    #for c in comp_topics.keys():
    #    comp_topics[c]['emb']=_norm_emb_probs(comp_topics[c]['emb'])
    nwords = len(uni_vocab)
    ntopics = [comp_topics[k]['emb'].shape[1] for k in comp_topics.keys()]
    uni_emb = np.memmap(
        _config['TRM']['uni_emb'],
        dtype='float32',
        mode='w+',
        shape=(sum(ntopics),nwords))
    emb_mask = np.memmap(
        _config['TRM']['emb_mask'],
        dtype='float32',
        mode='w+',
        shape=(sum(ntopics), nwords))
    itop = 0
    start_time = time.time()
    for ic, k in enumerate(comp_topics.keys()):
        comp = comp_topics[k]
        for w in comp['vocab']:
            if w not in uni_vocab:
                continue
            wuid = uni_vocab[w]
            local_wid = comp['vocab'][w]
            p_topic_given_word_local = comp['emb'][local_wid,:]
            # create a  (vocab x company_companyTopics) embedding matrix
            uni_emb[itop:itop + ntopics[ic], wuid] = p_topic_given_word_local
            emb_mask[itop:itop + ntopics[ic], wuid] = 1
        itop += ntopics[ic]
        print('{}/{}, {} secs'.format(ic, len(comp_topics), int(time.time()-start_time)))
    uni_emb.flush()
    emb_mask.flush()
    uni_emb = np.memmap(
        _config['TRM']['uni_emb'],
        dtype='float32',
        mode='c',
        shape=(sum(ntopics), nwords))
    emb_mask = np.memmap(
        _config['TRM']['emb_mask'],
        dtype='float32',
        mode='c',
        shape=(sum(ntopics), nwords))
    return uni_emb, emb_mask


class LocalTopicAsEmbedding():
    def __init__(self, config_file):
        self._config= read_config(config_file)
        if self._config['model']['name']=='topic_embedding_model':
            self.model= topic_embedding_model(config_file)
        if self._config['model']['name']=='fasttext_embedding_model':
            self.model= fasttext_embedding_model(config_file)
        self.uni_vocab={}
        self.comp_topics, self.global_topics= {}, {}
        self.trm= None
        self.uni_emb = []
        self.uni_emb_imp = []
        self.emb_mask = []
        self.cf=None
        self.imputor_config = self._config.get('Imputor', None)

    def _load_emb(self):
        if 'local' in self._config.keys():
            self.comp_topics = load(self._config['local']['embedding_path'])
        self.global_topics = load(self._config['global']['embedding_path'])
        self.uni_vocab = load( self._config['TRM']['uni_vocab'])
        nwords = len(self.uni_vocab)
        ntopics = [self.comp_topics[k]['emb'].shape[1] for k in self.comp_topics.keys()]
        self.uni_emb = np.memmap(
            self._config['TRM']['uni_emb'],
            dtype='float32',
            mode='c',
            shape=(sum(ntopics), nwords))
        imputed_emb_path = ''
        if self.imputor_config:
            imputed_emb_path = self.imputor_config.get('imputed_emb_path', '')
        if os.path.exists(imputed_emb_path) and \
                self.imputor_config.get('load_embedding_imp', False):
            self.uni_emb_imp = np.memmap(
                imputed_emb_path,
                dtype='float32',
                mode='c',
                shape=self.uni_emb.shape)
        print(self.uni_emb.shape)
        self.emb_mask = np.memmap(
            self._config['TRM']['emb_mask'],
            dtype='float32',
            mode='c',
            shape=(sum(ntopics), nwords))
        return

    def _load_trm(self):
        # only local trm is needed. Global TRM is only for comparison purpose.
        if self._config['TRM'].get('local', False):
            self.trm = load(self._config['TRM']['local']['model_path'])

    def _load(self):
        self._load_emb()
        self._load_trm()

    def train(self,df):
        df = df[['Company', "Text"]]
        if self._config['model']['load_embedding']:
            self._load_emb()
        else:
            self.comp_topics, self.global_topics, self.uni_vocab = \
                local_global_modeling(self.model, df, self._config)
            self.uni_emb, self.emb_mask = construct_universal_emb(
                self._config, self.comp_topics, self.uni_vocab
            )
        if self.imputor_config:
            if (not self._config.get('load_embedding_imp', False)) or (self.uni_emb_imp is None):
                self.cf = imputor(self.imputor_config, self.uni_emb, self.emb_mask)
                self.uni_emb_imp = impute(self.imputor_config, self.cf, self.uni_emb, self.emb_mask)
            return

        return

    def get_local_names(self):
        return [c for c in self.comp_topics.keys()]

    def get_tickers(self):
        return [self.comp_topics[c]['tic']
                for c in self.comp_topics.keys()]

    def predict(self, keywords, comp_list=None):
        keywords= _lem_stem(keywords)
        voc = list(self.uni_vocab.keys())
        wuid, keywords = zip(*[(self.uni_vocab[w], w) for w in keywords if w in voc])
        wuid, keywords = list(wuid), list(keywords)
        print('found keywords', keywords)
        print('getting the word unique embedding')
        search_wvecs = self.uni_emb[:, wuid]
        print('done.')
        #search_wvecs=[]
        #for i, id in enumerate(wuid):
        #    search_wvecs.append(self.uni_emb[:, id:id+1])
        #    print('%d / %d' % (i, len(wuid)))
        #search_wvecs = np.concatenate(search_wvecs, axis=0)
        comp_list = self.get_local_names() if comp_list is None else comp_list
        data = np.zeros((len(comp_list), len(keywords)))
        ntopics = [self.comp_topics[k]['emb'].shape[1] for k in comp_list]
        itopic = 0
        for i, c in enumerate(comp_list):
            search_cwvecs = search_wvecs[itopic:itopic + ntopics[i], :]
            cwvec = self.comp_topics[c]['emb']
            sim_d = self.model.dist_func(search_cwvecs.T, cwvec)
            data[i, :] = 1 - sim_d.min(1)
            itopic += ntopics[i]
            print(' - predict %d  / %d' % (i, len(comp_list)))
        return data, keywords

    def predict_trm(self, keywords):
        data = predict_with_trm(keywords,
                         _lem_stem,
                         self.trm,
                         self.uni_emb,
                         self.uni_vocab,
                         self.comp_topics,
                         self.get_local_names(),
                         self.model.dist_func )
        return data


class GlobalTopicAsEmbedding():
    def __init__(self, config_file):
        self._config= read_config(config_file)
        self.model= topic_embedding_model(config_file)
        self.comp_topics = {}
        self.global_topics = {}

    def _load(self):
        if 'local' in self._config.keys():
            self.comp_topics = load(self._config['local']['embedding_path'])
        self.global_topics = load(self._config['global']['embedding_path'])
        return

    def get_local_names(self):
        return [c for c in self.comp_topics.keys()]

    def get_tickers(self):
        return [self.comp_topics[c]['tic']
                for c in self.comp_topics.keys()]

    def predict(self, keywords, comp_list=None):
        # all embedding matrix should have columns represent topics, row represent words
        keywords = _lem_stem(keywords)
        model = self.global_topics
        wid, keywords = zip(*[(model['vocab'][w], w) for w in keywords \
                              if w in model['vocab'].keys()])
        print('found keywords', keywords)
        wid, keywords = list(wid), list(keywords)
        comp_list = self.get_local_names() if comp_list is None else comp_list
        data = np.zeros((len(comp_list), len(keywords)))
        search_wvecs = model['emb'][wid, :]
        for i, c in enumerate(comp_list):
            idx = [model['vocab'][k] for k in self.comp_topics[c]['vocab'].keys() \
                   if k in model['vocab'].keys()]
            wvecs = model['emb'][idx, :]
            sim = euclidean_distances(search_wvecs, wvecs)
            data[i, :] = 1 - sim.min(1)
            print(' - predict %d  / %d' % (i, len(comp_list)))
        return data, keywords
