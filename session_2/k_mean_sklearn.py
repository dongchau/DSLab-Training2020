from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import numpy as np


def load_data(data_path, vocab_path):
    """
    Load data from data_tf_idf.txt
    :param data_path: string representing path to data_tf_idf.txt
    :return:
    """
    def sparse_to_dense(sparse_r_d, vocab_size):
        """
        Convert tf-idf from sparse vector to dense vector
        :param sparse_r_d: String representing sparse vector form of tf-idf
        :param vocab_size: int value representing size of vocabulary
        :return: numpy array representing dense vector form of tf-idf
        """
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf
        return np.array(r_d)

    # read data and vocabulary from file
    with open(data_path) as f:
        d_lines = f.read().splitlines()
    with open(vocab_path) as f:
        vocab_size = len(f.read().splitlines())

    # standardize each line in data
    data = []
    labels = []
    for data_id, d in enumerate(d_lines):
        features = d.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        labels.append(label)
        r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
        data.append(r_d)

    return data, labels


def clustering_with_kmean(data_path, vocab_path):
    data, labels = load_data(data_path, vocab_path)
    X = csr_matrix(data)
    kmeans = KMeans(
        n_clusters=20,
        init='random',
        n_init=5,  # number of time that kmean runs with differently initialized centroids
        tol=1e-3,  # threshold to acceptable minimum error decrease
        random_state=2018  # set to get deterministic results
    ).fit(X)
    clustering_labels = kmeans.labels_
    return clustering_labels

#
# def compute_purity(labels):
#     majority_sum = 0
#     for cluster in self._clusters:
#         member_labels = [member.get_label() for member in cluster.get_members()]
#         max_count = max([member_labels.count(label) for label in range(20)])
#         majority_sum += max_count
#     return majority_sum * 1.0 / len(self._data)


data_path = "G:/Project/python/DSLab Training 2020/datasets/20news-bydate/data_tf_idf.txt"
vocab_path = "G:/Project/python/DSLab Training 2020/datasets/20news-bydate/words_idfs.txt"
print(clustering_with_kmean(data_path, vocab_path))
