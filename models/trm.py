import numpy as np
import time
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from itertools import cycle

class defaultTRM():
    def __init__(self, mat):
        self.labels_= list(range(mat.shape[0]))

def _compute_stats(data):
    d_stats = {
        'min': data.min(),
        'max': data.max(),
        'std': data.std(),
        'mean': data.mean()
    }
    #d_stats.update({
    #    str(i) + 'percentile': np.percentile(data, i) for i in range(10, 100, 10)
    #})
    return d_stats

def _precompute_csr_pdist(X, eps, chunk_size, dist_fn):
    nsamples, ndims = X.shape
    sqndims = np.sqrt(ndims)
    def _calc_dist_chunk(vec):
        ii = 0
        d = []
        for n in range(nsamples//chunk_size):
            #d.append(self.model.dist_func(vec, X[ii:ii+chunk_size, :]))
            #d.append(euclidean_distances(vec, X[ii:ii + chunk_size, :])/sqndims)
            di = dist_fn(vec, X[ii:ii + chunk_size, :])
            d.append(di)
            ii+= chunk_size
        if ii<nsamples:
            #d.append(self.model.dist_func(vec, X[ii:nsamples, :]))
            #d.append(euclidean_distances(vec, X[ii:nsamples, :])/sqndims)
            d.append(dist_fn(vec, X[ii:nsamples, :]))
        return np.hstack(d)

    def _reduce_func(D_chunk):
        dists = _calc_dist_chunk(D_chunk)
        #stats = _compute_stats(dists)
        #print(stats)
        dists = np.maximum(0, dists)
        neigh = np.flatnonzero(dists < eps)
        return neigh,dists[0,neigh]

    row_ind = []
    col_ind = []
    neighbors = []
    start_time= time.time()
    i=0
    for i, xx in enumerate(X):
        neigh, values= _reduce_func(np.atleast_2d(xx))
        col_ind+= neigh.tolist()
        row_ind +=[i for _ in range(len(neigh))]
        neighbors += values.tolist()
        if (i+1)%1000==0:
            print('%d/%d, %d secs' % (
            i+1, nsamples, int(time.time()-start_time) ))
    print('%d/%d, %d secs' % (
        i + 1, nsamples, int(time.time() - start_time)))
    print('done.')
    return csr_matrix(
                (neighbors, (row_ind, col_ind)),
                shape=(nsamples, nsamples)
            )


def dbscan_(eps, min_samples, pdist_chunk_size, pdist_emb, dist_fn):
    pdist = _precompute_csr_pdist(
        pdist_emb,
        eps=eps,
        chunk_size=pdist_chunk_size,
        dist_fn=dist_fn
    )
    d_stats = _compute_stats(pdist.data)
    ndata=len(pdist.data)
    ntotal = pdist.shape[0]*pdist.shape[1]
    d_stats.update({'num_pdist':ntotal,
                    'pdist_sparsity_ratio': 1-ndata/ntotal
                    })
    print('dbscan distance stats - ')
    for k,v in d_stats.items():
        print(k,':',v)
    #save(pdist, pdist_path)
    model = DBSCAN(
        metric="precomputed",
        min_samples= min_samples,
        eps= eps
    ).fit(pdist)
    return model


def get_uni_trm_centers(trm, uni_emb):
    uni_topic_repr_vec = []
    uni_label = sorted(set(trm.labels_) - {-1})
    for l in uni_label:
        uni_topic_repr_vec.append(
            np.atleast_2d(uni_emb[trm.labels_ == l,:].mean(0))
        )
    singleton_ind = np.flatnonzero(trm.labels_ == -1)
    trm_labels = trm.labels_
    if not singleton_ind:
        return uni_topic_repr_vec, trm_labels
    m = max(uni_label)+1
    trm_labels[singleton_ind] = range(m, m + len(singleton_ind))
    for i in range(m, m + len(singleton_ind)):
        uni_topic_repr_vec.append( uni_emb[trm_labels==i,:])
    uni_topic_repr_vec = np.vstack(uni_topic_repr_vec)
    return uni_topic_repr_vec, trm_labels




all_color_names= list(mcolors.TABLEAU_COLORS.keys())\
                + list(mcolors.BASE_COLORS.keys())
#+ list(mcolors.CSS4_COLORS.keys())
colormap = cycle( set(all_color_names)-{'k','black','w','white'})

def plot_tsne(features, labels,vocab, sel_nlabels=None,title='tsne of word-topic embedding',hold=True):
    if sel_nlabels is not None:
        ls = np.arange(0, max(labels) + 1)
        np.random.shuffle(ls)
        sel_ind = np.hstack([np.flatnonzero(labels==l) for l in [-1]+ls[:sel_nlabels].tolist()])
        labels = labels[sel_ind]
        data = features[sel_ind,:]
    else:
        data = features
    tsne = TSNE(n_components=2).fit_transform(data)
    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in set(labels):
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]
        ind = np.argsort(features[indices,:],axis=1)[:,-2:]
        words = list(set(sum( [vocab[ids].tolist() for ids in ind], [])))[:4]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format
        #color = np.array(colors_per_class[label], dtype=np.float) / 255
        if label==-1:
            color='k'
        else:
            color = next(colormap)
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color, s=0.8,label=label)
        ax.annotate(','.join(words), (current_tx[0], current_ty[0]))

    # build a legend using the labels we set previously
    ax.legend(loc='best')
    ax.set_title(title)
    # finally, show the plot
    plt.show(block=hold)
    return plt


def fit_dbscan_as_TRM(_config, local_topics,  global_topics, dist_func, save_fn):
    dbscan_global = dbscan_(
            eps= _config['TRM']['global']['eps'],
            min_samples=_config['TRM']['global']['min_samples'],
            pdist_chunk_size=_config['TRM']['pdist_chunk_size'] ,
            pdist_emb= global_topics['emb'].transpose(),
            dist_fn= dist_func
        )
    print(len(set(dbscan_global.labels_)) - 1, ' different global clusters')
    print('global singleton topic rate: ', (dbscan_global.labels_ == -1).mean())
    save_fn(dbscan_global, _config['TRM']['global']['model_path'])
    if 'local' not in _config.keys():
        return
    trm = dbscan_(
        eps= _config['TRM']['local']['eps'],
        min_samples= _config['TRM']['local']['min_samples'],
        pdist_chunk_size=_config['TRM']['pdist_chunk_size'] ,
        pdist_emb= local_topics['uni_emb'],
        dist_fn = dist_func
    )
        #trm= KMeans(n_clusters=200).fit(uni_emb.transpose())
    print(len(set(trm.labels_))-1,' different clusters')
    print('singleton topic rate: ',(trm.labels_==-1).mean())
    plot_tsne(global_topics['emb'],
                dbscan_global.labels_,
                vocab=np.array( list(global_topics['vocab'].keys())),
                title='tsne of global VAE Topic word-topic emb',
                hold=False)
    plot_tsne(local_topics['uni_emb'],
                trm.labels_,
                sel_nlabels=5,
                vocab=np.array( list(local_topics['uni_vocab'].keys())),
                title='tsne of VAE Topic Relational word-topic emb',
                hold=True)
    save_fn(trm, _config['TRM']['local']['model_path'])
    return trm


def predict_with_trm(keywords, preproc_fn, trm, uni_emb,uni_vocab, comp_topics, comp_list, dist_func):
    # all embedding matrix should have columns represent topics, row represent words
    #keywords = _lem_stem(keywords)
    keywords = preproc_fn(keywords)
    comp_topics = comp_topics.copy()
    #for c in comp_topics.keys():
    #    comp_topics[c]['emb'] = _norm_emb_probs(comp_topics[c]['emb'])
    uni_trm_centers, trm_labels = get_uni_trm_centers(trm, uni_emb)
    voc = list(uni_vocab.keys())
    wuid, keywords = zip(*[(uni_vocab[w], w) for w in keywords if w in voc])
    wuid, keywords = list(wuid), list(keywords)
    print('found keywords', keywords)
    search_wvecs = uni_trm_centers[:,wuid]
    #comp_list = self.get_local_names()
    data = np.zeros((len(comp_list), len(keywords)))
    #data_0 = np.zeros((len(comp_list), len(keywords)))
    ntopics = [comp_topics[k]['emb'].shape[1] for k in comp_list]
    itopic = 0
    for i, c in enumerate(comp_list):
        cwvec = []
        topic_clusters = trm_labels[itopic:itopic + ntopics[i]]
        labels = sorted(set(topic_clusters))
        search_cwvecs = search_wvecs[labels,:]
        # unify the wordvec repr. dimensions using related topic labeled centers.
        for l in labels:
            prob_vec = np.atleast_2d(comp_topics[c]['emb'][:, l == topic_clusters]).mean(1)
            cwvec.append(prob_vec)
        cwvec = np.stack(cwvec)
        print(itopic, ntopics[i], len(cwvec))
        if len(cwvec)==0:
            data[i, :] = 0
            itopic += ntopics[i]
            continue
        # search_cwecs - (num_of_keywords)x (num_of_topic clusters for company c)
        # cwvec - (num_of words for company c) x (num_of_topic clusters for company c)
        # similarity of the average company topic embedding vector to the AVG. universal topic embedding vector
        #sim_d = self.model.dist_func(search_cwvecs.T, cwvec.T)
        sim_d = dist_func(search_cwvecs.T, cwvec.T)
        data[i, :] = 1 - sim_d.min(1)
        #data[i, :] = (1 - sim_kl.min(1)) / 10 + 1 - sim_d.min(1)
        #data_0[i, :] = 1 - sim_d.min(1)
        itopic += ntopics[i]
    return data