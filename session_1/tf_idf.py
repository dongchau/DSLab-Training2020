import numpy as np
import os
from nltk.stem.porter import PorterStemmer
import re
from collections import defaultdict
from nltk.corpus import stopwords


def gather_20newsgroups_data():
    """
    Thu thập và chuẩn hóa dữ liệu
    :return:
    """
    # đường dẫn đến thư mục chứa dữ liệu
    path = 'datasets/20news-bydate/'
    # dirs = [path + dir_name + '/' for dir_name in os.listdir(path) if not os.path.isfile(path + dir_name)]
    # train_dir, test_dir = (dirs[0], dirs[1] if 'train' in dirs[0] else (dirs[1], dirs[0]))
    train_dir, test_dir = (path + '20news-bydate-train', path + '20news-bydate-test')
    list_newsgroups = [newsgroup for newsgroup in os.listdir(train_dir)]
    list_newsgroups.sort()

    # nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stemmer = PorterStemmer()

    def collect_data_from(parent_dir, newsgroup_list):
        """
        Đọc và xử lí dữ liệu
        :param parent_dir: 1 xâu biểu diễn đường dẫn tới thư mục chứa dữ liệu
        :param newsgroup_list: 1 list biểu diễn danh sách thư mục con
        :return: 1 list dữ liệu đã được chuẩn hóa, mỗi phần tử của list là một
            xâu có dạng "nhãn lớp <fff> tên file chứa dữ liệu <fff> nội dung văn bản"
        """
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            label = group_id
            dir_path = parent_dir + '/' + newsgroup + '/'
            files = [(filename, dir_path + filename)
                     for filename in os.listdir(dir_path)
                     if os.path.isfile(dir_path + filename)]
            files.sort()
            for filename, filepath in files:
                with open(filepath) as f:
                    # đọc dữ liệu từ file
                    text = f.read().lower()
                    # loại bỏ stop words và đưa các từ về dạng nguyên thủy
                    words = [stemmer.stem(word)
                             for word in re.split('\W+', text)
                             if word not in stop_words]
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' + filename + '<fff>' + content)
        return data

    train_data = collect_data_from(parent_dir=train_dir, newsgroup_list=list_newsgroups)
    test_data = collect_data_from(parent_dir=test_dir, newsgroup_list=list_newsgroups)

    # xuất dữ liệu ra file
    full_data = train_data + test_data
    with open('datasets/20news-bydate/20news-train-processed.txt', 'w') as f:
        f.write('\n'.join(train_data))

    with open('datasets/20news-bydate/20news-test-processed.txt', 'w') as f:
        f.write('\n'.join(test_data))

    with open('datasets/20news-bydate/20news-full-processed.txt', 'w') as f:
        f.write('\n'.join(full_data))


def generate_vocabulary(data_path):
    """
    Tạo từ điển
    :param data_path: đường dẫn đến file văn bản
    :return: ghi file từ điển ra file
    """
    def compute_idf(df, corpus_size):
        """
        Tính idf của từ x
        :param df: tổng số văn bản có chứa từ x
        :param corpus_size: tổng số văn bản trong tập dữ liêu
        :return: idf của từ x
        """
        assert df > 0
        return np.log10(corpus_size * 1.0 / df)

    # đọc dữ liệu từ file và đếm số văn bản
    with open(data_path) as f:
        lines = f.read().splitlines()
    doc_count = defaultdict(int)
    corpus_size = len(lines)

    # ghi lại các từ trong văn bản và số lần xuất hiện của chúng
    for line in lines:
        features = line.split('<fff>')
        text = features[-1]
        words = list(set(text.split()))
        for word in words:
            doc_count[word] += 1

    # tính idf của các từ xuất hiện trong trên 10 văn bản
    words_idfs = [(word, compute_idf(document_freq, corpus_size))
                  for word, document_freq in zip(doc_count.keys(), doc_count.values())
                  if document_freq > 10 and not word.isdigit()]
    words_idfs.sort(key=lambda word_idf: -word_idf[1])
    print('Vocabulary size: {}'.format(len(words_idfs)))
    with open('datasets/20news-bydate/words_idfs.txt', 'w') as f:
        f.write('\n'.join([word + '<fff>' + str(idf) for word, idf in words_idfs]))


def get_tf_idf(data_path):
    """
    Tính tf-idf của từng từ trong từ điển
    :param data_path: đường dẫn tới file từ điển
    :return: tạo file chứa các từ trong từ điển và giá trị tf-idf tương ứng của từ đó
    """
    # đọc từ điển từ file
    with open('datasets/20news-bydate/words_idfs.txt') as f:
        words_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1]))
                      for line in f.read().splitlines()]
        word_IDs = dict([(word, index) for index, (word, idf) in enumerate(words_idfs)])
        idfs = dict(words_idfs)

    with open(data_path) as f:
        documents = [(int(line.split('<fff>')[0]),
                     int(line.split('<fff>')[0]),
                     line.split('<fff>')[2])
                     for line in f.read().splitlines()]

    data_tf_idf = []
    for document in documents:
        label, doc_id, text = document
        words = [word for word in text.split() if word in idfs]
        word_set = list(set(words))
        max_term_freq = max([words.count(word) for word in word_set])
        words_tfidfs = []
        sum_squares = 0.0
        # tính tf-idf cho từng từ
        for word in word_set:
            term_freq = words.count(word)
            tf_idf_value = term_freq * 1.0 / max_term_freq * idfs[word]
            words_tfidfs.append((word_IDs[word], tf_idf_value))
            sum_squares += tf_idf_value ** 2

        # chuẩn hóa từng chiều của vector đặc trưng cho từng văn bản
        # về đoạn [0, 1]
        words_tfidfs_normalized = [str(index) + ':' +
                                   str(tf_idf_value / np.sqrt(sum_squares))
                                   for index, tf_idf_value in words_tfidfs]

        sparse_rep = ' '.join(words_tfidfs_normalized)
        data_tf_idf.append((label, doc_id, sparse_rep))

    # ghi dữ liệu ra file
    with open('datasets/20news-bydate/data_tf_idf.txt', 'w') as f:
        f.write('\n'.join([str(label) + '<fff>' + str(doc_id) + '<fff>' + sparse_rep
                           for label, doc_id, sparse_rep in data_tf_idf]))


if __name__ == '__main__':
    gather_20newsgroups_data()
    generate_vocabulary('datasets/20news-bydate/20news-train-processed.txt')
    get_tf_idf('datasets/20news-bydate/20news-train-processed.txt')
