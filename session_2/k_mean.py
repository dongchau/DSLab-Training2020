from collections import defaultdict
import numpy as np
import sklearn


class Member:
    """
    Lưu trữ thông tin của đối tượng
    """
    def __init__(self, r_d, label=None, doc_id=None):
        """
        Constructor
        :param r_d: biểu diễn tf-idf của văn bản d
        :param label: newsgroup của văn bản d
        :param doc_id: file chứa văn bản d
        """
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id

    def get_r_d(self):
        return self._r_d

    def get_label(self):
        return self._label

    def get_doc_id(self):
        return self._doc_id


class Cluster:
    """Thông tin về 1 cụm"""
    def __init__(self):
        # Centroid of cluster
        self._centroid = None
        # List of cluster member
        self._members = []

    def get_members(self):
        return self._members

    def get_centroid(self):
        return self._centroid

    def set_centroid(self, new_centroid):
        self._centroid = new_centroid

    def reset_members(self):
        self._members = []

    def add_member(self, member):
        self._members.append(member)


class Kmean:
    def __init__(self, num_cluster):
        """
        Constructor
        :param num_cluster: int value representing the number of clusters
        """
        self._num_clusters = num_cluster
        self._clusters = [Cluster() for _ in range(num_cluster)]
        # list of centroid
        self._E = []
        # overall similarity
        self._S = 0
        # list of all data point
        self._data = []
        # number of member in each label
        self._label_count = defaultdict(int)
        # number of iteration
        self._iteration = 0
        # new overall similarity
        self._new_S = 0

    def load_data(self, data_path):
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
        with open("G:/Project/python/DSLab Training 2020/datasets/20news-bydate/words_idfs.txt") as f:
            vocab_size = len(f.read().splitlines())

        # standardize each line in data
        for data_id, d in enumerate(d_lines):
            features = d.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
            self._data.append(Member(r_d=r_d, label=label, doc_id=doc_id))

    def random_init(self):
        """
        Randomly pick self._num_clusters point from self._data as initial centroid
        :return:
        """
        members = [member.get_r_d() for member in self._data]
        pos = np.random.choice(len(self._data), self._num_clusters, replace=False)
        centroid = []
        for i in pos:
            centroid.append(members[i])
        self._E = centroid
        for i in range(self._num_clusters):
            self._clusters[i].set_centroid(centroid[i])

    @staticmethod
    def compute_similarity(member, centroid):
        """
        Calculate cosine similarity between member and centroid of cluster
        :param member: Member object representing a data point
        :param centroid: 1-dimensional matrix representing coordinate of centroid of cluster
        :return: cosine similarity between member and centroid of cluster
        """
        return np.dot(centroid, member.get_r_d())

    @staticmethod
    def compute_distance(member, centroid):
        """
        Calculate Euclid distance between member and centroid of cluster
        :param member: Member object representing a data point
        :param centroid: 1-dimensional matrix representing coordinate of centroid of cluster
        :return: Euclid distance between member and centroid of cluster
        """
        return np.linalg.norm(centroid - member.get_r_d())

    def select_cluster(self, member, loss):
        """
        Select cluster for each member in dataset
        :param member: Member object representing a data point
        :param loss: String representing method to calculate
            similarity between member and centroid:
            loss='similarity': use cosine similarity
            loss='distance': use Euclid distance
        :return:
        """
        best_fit_cluster = None
        min_distance = 1000000000
        max_similarity = -1
        assert loss in ['distance', 'similarity']
        # calculate similarity between member and centroid of each cluster
        # and select cluster which has maximum similarity (or minimum distance)
        for cluster in self._clusters:
            if loss == 'distance':
                distance = self.compute_distance(member, cluster.get_centroid())
                if distance < min_distance:
                    best_fit_cluster = cluster
                    min_distance = distance
            else:
                similarity = self.compute_similarity(member, cluster.get_centroid())
                if similarity > max_similarity:
                    best_fit_cluster = cluster
                    max_similarity = similarity
        best_fit_cluster.add_member(member)
        if loss == 'distance':
            return min_distance
        else:
            return max_similarity

    @staticmethod
    def update_centroid(cluster):
        """
        Calculate centroid of each cluster
        :param cluster: Cluster object
        :return:
        """
        # get coordinate of all points in cluster
        member_r_ds = [member.get_r_d() for member in cluster.get_members()]
        aver_r_d = np.mean(member_r_ds, axis=0)
        # normalize centroid vector
        aver_r_d_length = np.sqrt(np.sum(aver_r_d ** 2))
        new_centroid = np.array([(value / aver_r_d_length) for value in aver_r_d])
        cluster.set_centroid(new_centroid)

    def stopping_condition(self, criterion, threshold):
        """
        :param criterion: String representing stopping criteria
        :param threshold:
        :return: True if
        """
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria
        if criterion == 'max_iters':
            self._S = self._new_S
            self._new_S = 0
            # return True if the number of iterators is greater than threshold
            if self._iteration >= threshold:
                return True
            else:
                return False
        elif criterion == 'similarity':
            S_update = abs(self._S - self._new_S)
            self._S = self._new_S
            self._new_S = 0
            # return True if total similarity change less than threshold
            if S_update <= threshold:
                return True
            else:
                return False
        else:
            E_new = []
            E_new_update = []
            self._new_S = 0
            for cluster in self._clusters:
                centroid = cluster.get_centroid()
                E_new.append(centroid)
                if not(any(np.equal(self._E, centroid).all(1))):
                    E_new_update.append(centroid)
            self._E = E_new
            # return True if two sets of centroid are the same
            if len(E_new_update) <= 0:
                return True
            else:
                return False

    def run(self, criterion, threshold):
        self.random_init()
        # continually update clusters until convergence
        while True:
            for cluster in self._clusters:
                cluster.reset_members()
            for member in self._data:
                self._new_S += self.select_cluster(member, loss='similarity')
            for cluster in self._clusters:
                self.update_centroid(cluster)
            self._iteration += 1
            if self.stopping_condition(criterion, threshold):
                break

        print('Total iteration: ', self._iteration)

    def compute_purity(self):
        """
        Calculate Purity value
        :return: purity value
        """
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member.get_label() for member in cluster.get_members()]
            # get class which has largest number of labels in cluster
            max_count = max([member_labels.count(label) for label in range(20)])
            majority_sum += max_count
        return majority_sum * 1.0 / len(self._data)

    def compute_NMI(self):
        """
        Calculate NMI value
        :return: NMI value
        """
        I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)
        # calculate I(omega, C), H(omega) value
        for cluster in self._clusters:
            wk = len(cluster.get_members()) * 1.0
            H_omega += -wk / N * np.log10(wk / N)
            member_labels = [member.get_label() for member in cluster.get_members()]
            for label in range(20):
                wk_cj = member_labels.count(label) * 1.0
                cj = self._label_count[label]
                I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
        # calculate H(C) value
        for label in range(20):
            cj = self._label_count[label] * 1.0
            H_C += - cj / N * np.log10(cj / N)
        return I_value * 2.0 / (H_omega * H_C)


if __name__ == "__main__":
    kmean = Kmean(20)
    kmean.load_data("G:/Project/python/DSLab Training 2020/datasets/20news-bydate/data_tf_idf.txt")
    # run with max iteration stopping criteria
    kmean.run('max_iters', threshold=20)
    print("Max iterator purity: ", kmean.compute_purity())
    print("Max iterator NMI: ", kmean.compute_NMI())
    # run with total similarity stopping criteria
    kmean.run('similarity', threshold=1)
    print("Total similarity purity: ", kmean.compute_purity())
    print("Total similarity NMI: ", kmean.compute_NMI())
    # run with centroid stopping criteria
    kmean.run('centroid', threshold=0)
    print("Centroid purity: ", kmean.compute_purity())
    print("Centroid NMI: ", kmean.compute_NMI())
