from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import warnings
from multiprocessing import Process
import numpy as np
import argparse
import os
from util.util import set_seed

warnings.filterwarnings("ignore")
set_seed()
parser = argparse.ArgumentParser(
    description='Data pattern graph generator')
# Add arguments
parser.add_argument(
    '-d', '--dataset_name', type=str, help='dataset name (opportunity or realdisp)', default='opportunity')
parser.add_argument(
    '-l', '--load_path', type=str, help='dataset name (opportunity or realdisp)', default='opportunity_24')
parser.add_argument(
    '-p', '--placement', type=str, help='dataset name (opportunity or realdisp)', default='ideal')
# parser.add_argument('-s','--save_path', type=str, help='dataset name (opportunity or realdisp)', default='opportunity_24')
cfg = parser.parse_args()
if cfg.dataset_name == 'opportunity_5imu':
    LOAD_PATH = os.path.join('data/dataset/opportunity', cfg.load_path,'train_data.npz')
    DEVICE_NUM = 3 * 5
    SAVE_PATH = os.path.join('data/dataset/opportunity', cfg.load_path)
else:
    raise NotImplementedError(
        "Dataset {} is not supported".format(cfg.dataset))

def z_scoreLoader(path):
    train_data = np.load(path)
    train_x, train_y = train_data['data'], train_data['target']
    return train_x


class LDA:
    def __init__(self, K, alpha, beta, docs, V, smartinit=True):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.docs = docs
        self.V = V
        self.z_m_n = []
        self.n_m_z = np.zeros((len(self.docs), K)) + alpha   # (19, 20)
        self.n_z_t = np.zeros((K, V)) + beta   # (20, 4761)
        self.n_z = np.zeros(K) + V * beta   # (20)
        self.N = 0
        for m, doc in enumerate(docs):  # index and value
            self.N += len(doc)   # += 4761
            z_n = []
            for t in doc:
                if smartinit:
                    p_z = self.n_z_t[:, t] * self.n_m_z[m] / self.n_z
                    z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                else:
                    z = np.random.randint(0, K)
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
            self.z_m_n.append(np.array(z_n))
            # print("LDA done")

    def inference(self):   # """learning once iteration"""
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            n_m_z = self.n_m_z[m]
            for n, t in enumerate(doc):
                # discount for n-th word t with topic z
                z = z_n[n]
                n_m_z[z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1
                # sampling topic new_z for t
                p_z = self.n_z_t[:, t] * n_m_z / self.n_z
                new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                # set z the new topic and increment counters
                z_n[n] = new_z
                n_m_z[new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def worddist(self):
        """get topic-word distribution"""
        return self.n_z_t / self.n_z[:, np.newaxis]

    def perplexity(self, docs=None):
        if docs == None:
            docs = self.docs
        phi = self.worddist()   # (20, 4761)
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha)
            for w in doc:
                log_per -= np.log(np.inner(phi[:, w], theta))
            N += len(doc)
        per = np.exp(log_per / N)
        return per


def lda_learning(lda, iteration):
    pre_perp = lda.perplexity()
    # print("initial perplexity=%f" % pre_perp)
    for i in range(iteration):
        lda.inference()
        perp = lda.perplexity()
        # print("-%d p=%f" % (i + 1, perp))
        if pre_perp:
            if pre_perp < perp:
                output_word_topic_dist(lda)
                pre_perp = None
            else:
                pre_perp = perp
    output_word_topic_dist(lda)
    doc_topic = lda.n_m_z
    return doc_topic


def output_word_topic_dist(lda):
    zcount = np.zeros(lda.K, dtype=int)
    # length 20, each element is a dict, saves the frequency of each appeared word
    wordcount = [dict() for k in range(lda.K)]
    for xlist, zlist in zip(lda.docs, lda.z_m_n):
        for x, z in zip(xlist, zlist):
            zcount[z] += 1
            if x in wordcount[z]:
                wordcount[z][x] += 1
            else:
                wordcount[z][x] = 1

    phi = lda.worddist()   # topic-word distribution (20, 4761)
    return phi
    # for k in range(lda.K):
    #     print("\n-- topic: %d (%d words)" % (k, zcount[k]))
    #     for w in np.argsort(-phi[k])[:20]:
    #         print("%f (%d)" % (phi[k, w], wordcount[k].get(w, 0)))


def windows(load_path):
    x_all = z_scoreLoader(load_path)
    print("all graph shape", x_all.shape)  # (16122, 24, 105)
    all_win = np.split(x_all, DEVICE_NUM, axis=-1)
    all_win = np.concatenate(all_win, axis=0)
    print("all windows shape", all_win.shape)  # (564270, 24, 3)
    win_num = len(all_win)
    new_win = []
    pca = PCA(n_components=1)
    for i in range(win_num):
        data = all_win[i]    # (24, 3)
        # (24, 1)  normalized graph is too small to be divisor
        reduce = pca.fit_transform(data)
        reduce_list = list(reduce)
        new_win.append(reduce_list)
    temp = len(new_win)
    print("pca windows length", temp)  # 564270
    print("windows end")
    return new_win, temp


def cluster(data, clusters, mini_batch):
    # print("cluster start")
    # kmeans = KMeans(init='k-means++', n_clusters=clusters, n_init=10)
    # kmeans.fit_predict(graph)
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusters,
                          batch_size=mini_batch, n_init=10, max_no_improvement=10, verbose=0)
    mbk.fit_predict(data)
    output = mbk.labels_          # 564270
    # print("clustering output", output)
    return output


def dtw(doc):
    adj = np.zeros((DEVICE_NUM, DEVICE_NUM), dtype=np.float32)
    for i in range(DEVICE_NUM):
        for j in range(DEVICE_NUM):
            distance, path = fastdtw(doc[i], doc[j], dist=euclidean)
            adj[i][j] = distance
    return adj


def process_data(n_cluster, n_topic, docs, vocas):
    print(n_cluster, n_topic)
    # doc = sensor, topic = 指定参数, word = mbk_result
    lda = LDA(K=n_topic, alpha=0.1, beta=0.01,
              docs=docs, V=vocas, smartinit=False)
    doc_topic = lda_learning(lda, 100)
    print("=========================================")
    print(doc_topic.shape)
    adj = cosine_similarity(doc_topic)  # (35, 35)
    for p in range(adj.shape[0]):
        adj[p][p] = 0
    np.savetxt(SAVE_PATH +
               '/pattern_graph_{}_{}.txt'.format(n_cluster, n_topic), adj, delimiter="\t")


if __name__ == "__main__":
    topics = [64]
    clusters = [64]
    data, win_num = windows(LOAD_PATH)         # list (90459, 24, 1)   564270
    data_list = np.array(data)   # (90459, 24, 1)   (564270, 24, 1)
    # (90459, 24)   (564270, 24)
    data_array = data_list.reshape((win_num, 24))
    p_list = []
    for n_cluster in clusters:
        print(n_cluster)
        words = cluster(data_array, n_cluster, 100)
        docs = words.reshape((DEVICE_NUM, -1))
        vocas = len(docs[0])
        for n_topic in topics:
            p = Process(target=process_data, args=(
                n_cluster, n_topic, docs, vocas,))
            p_list.append(p)
            p.start()
        # if len(p) >= 19:
        #     for p in p_list:
        #         p_list = []
        #         p.join()
    for p in p_list:
        p.join()
    # clusters = [64]
    # data, win_num = windows(LOAD_PATH)         # list (90459, 24, 1)   564270
    # data_list = np.array(data)   # (90459, 24, 1)   (564270, 24, 1)
    # data_array = data_list.reshape((win_num, 24))    # (90459, 24)   (564270, 24)
    # for n_cluster in clusters:
    #     print(n_cluster)
    #     words = cluster(data_array, n_cluster, 100)
    #     docs = words.reshape((DEVICE_NUM, -1))
    #     adj = dtw(docs)
    #     np.savetxt(SAVE_PATH + '/pattern_graph_dtw_{}.txt'.format(n_cluster), adj, delimiter="\t")