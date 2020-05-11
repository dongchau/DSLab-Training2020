from sklearn.svm import LinearSVC, SVC
from k_mean_sklearn import load_data
from sklearn.metrics import accuracy_score


def classinfying_with_linear_SVMs(train_path, test_path, vocab_path, model='SVC'):
    train_X, train_Y = load_data(train_path, vocab_path)
    assert model in ['SVC', 'Kernel-SVM']
    if model == 'SVC':
        classifier = LinearSVC(
            C=10.0,  # penalty coeff
            tol=0.001,  # tolerance for stopping criteria
            verbose=True  # whethers print out logs or not
        )
    else:
        classifier = SVC(
            C=50.0,
            kernel='rbf',
            gamma=0.1,
            tol=0.001,
            verbose=True
        )
    classifier.fit(train_X, train_Y)

    test_X, test_Y = load_data(test_path, vocab_path)
    predicted_y = classifier.predict(test_X)
    acc = accuracy_score(test_Y, predicted_y)
    print(acc)


train = "G:/Project/python/DSLab Training 2020/datasets/20news-bydate/data_tf_idf.txt"
test = "G:/Project/python/DSLab Training 2020/datasets/20news-bydate/test_tf_idf.txt"
vocab = "G:/Project/python/DSLab Training 2020/datasets/20news-bydate/words_idfs.txt"
classinfying_with_linear_SVMs(train, test, vocab, model='Kernel-SVM')
