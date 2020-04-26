import numpy as np


def load_data(data_path):
    """
    Đọc dữ liệu từ file death_rates-data.txt
    :param data_path: đường dẫn đến file
    :return:
        X: mảng 2 chiều biểu diễn tập thuộc tính
        mỗi hàng tương ứng với 1 điểm dữ liệu
        Y: mảng 1 chiều biểu diễn các giá trị
        death rate tương ứng với từng điểm dữ liệu
    """
    with open(data_path) as file:
        data = np.loadtxt(file)
        X = data[:, 1:16]
        Y = data[:, 16:]
    return X, Y


def normalize_and_add_ones(X):
    """
    Chuẩn hóa dữ liệu sử dụng feature scaling
    :param X: Tập thuộc tính đầu vào
    :return: tập thuộc tính đã chuẩn hóa và thêm vector cột [1 1 ... 1] vào
        đầu (để nhân với bias w0)
    """
    X_max = np.array([[np.amax(X[:, column_id])
                       for column_id in range(X.shape[1])]
                      for _ in range(X.shape[0])])
    X_min = np.array([[np.amin(X[:, column_id])
                       for column_id in range(X.shape[1])]
                      for _ in range(X.shape[0])])

    X_normalized = (X - X_min) / (X_max - X_min)

    ones = np.array([[1] for _ in range(X.shape[0])])
    return np.column_stack((ones, X_normalized))


class RidgeRegression:
    def __init__(self):
        return

    def fit(self, X_train, Y_train, LAMBDA):
        """
        Tìm nghiệm bằng công thức nghiệm 𝑤∗=(𝑋^𝑇 * 𝑋 + 𝜆 * 𝐼_(𝐾+1))^(−1) * 𝑋𝑌
        :param X_train: tập thuộc tính
        :param Y_train: tập giá trị đầu ra tương ứng với từng vector thuộc tính
        :param LAMBDA: weight decay, siêu tham số, 0 < LAMBDA < 1
        :return: vector tham số W
        """
        assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]
        W = np.linalg.inv(np.transpose(X_train).dot(X_train) + \
                          LAMBDA * np.identity(X_train.shape[1])
                          ).dot(np.transpose(X_train)).dot(Y_train)
        return W

    def fit_gradient_descent(self, X_train, Y_train, LAMBDA, lr, max_num_epochs=100, batch_size=128):
        """
        Tìm nghiệm bằng thuật toán gradient
        :param X_train: tập thuộc tính
        :param Y_train: tập giá trị đầu ra tương ứng với từng vector thuộc tính
        :param LAMBDA: weight decay, siêu tham số
        :param lr: learning rate, siêu tham số
        :param max_num_epochs: số lượng epoch khi train
        :param batch_size: kích thước batch
        :return: vector tham số W
        """
        W = np.random.rand(X_train.shape[1])
        W = W.reshape((X_train.shape[1], 1))
        lass_lost = 10e+8
        for epoch in range(max_num_epochs):
            # xáo trộn tập dữ liệu khi bắt đầu 1 epoch mới
            arr = np.array(range(X_train.shape[0]))
            np.random.shuffle(arr)
            X_train = X_train[arr]
            Y_train = Y_train[arr]
            # số batch trên 1 epoch
            total_minibatch = int(np.ceil(X_train.shape[0] / batch_size))
            for i in range(total_minibatch):
                index = i * batch_size
                X_train_sub = X_train[index:index + batch_size]
                Y_train_sub = Y_train[index:index + batch_size]
                # tính gradient của hàm loss
                grad = np.transpose(X_train_sub).dot(X_train_sub.dot(W) - Y_train_sub) + LAMBDA * W
                # cập nhật trọng số W
                W = W - lr * grad
            # điều kiện dừng: sau mỗi epoch nếu giá trị thay đổi của hàm loss
            # không quá 10^-5 thì kết thúc quá trình train
            new_loss = self.compute_RSS(self.predict(W, X_train), Y_train)
            if np.abs(new_loss - lass_lost) <= 1e-5:
                break
            lass_lost = new_loss
        return W

    def predict(self, W, X_new):
        """
        Đưa ra dự đoán gái trị death rate Y ứng với trọng số W và đầu vào X
        :param W: trọng số của mô hình
        :param X_new: giá trị input
        :return: giá trị dự đoán của Y
        """
        return np.array(X_new).dot(W)

    def compute_RSS(self, Y_new, Y_predicted):
        """
        Tính giá trị hàm loss
        :param Y_new: giá trị Y thực tế
        :param Y_predicted: Giá trị Y dự đoán bởi mô hình
        :return: Giá trị hàm MSE loss
        """
        return 1.0 / Y_new.shape[0] * np.sum((Y_new - Y_predicted) ** 2)

    def get_the_best_LAMBDA(self, X_train, Y_train):
        """
        Lựa chọn tham số Lambda bằng phương pháp k-fold cross validation
        :param X_train, Y_train: Tập các điểm dữ liệu huấn luyện và nhãn
        :return: giá trị Lambda làm cho hàm loss đạt giá trị nhỏ nhất trên
            tập huấn luyện
        """

        def cross_validation(num_folds, LAMBDA):
            # chia tập dữ liệu huấn luyện thành k phần
            row_ids = np.array(range(X_train.shape[0]))
            valid_ids = np.split(row_ids[:(len(row_ids) - len(row_ids) % num_folds)], num_folds)
            valid_ids[-1] = np.append(valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds:])
            train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
            aver_RSS = 0
            for i in range(num_folds):
                # với mỗi fold lấy 1 phần làm tập val. k - 1 phần làm tập train
                valid_part = {'X': X_train[valid_ids[i]], 'Y': Y_train[valid_ids[i]]}
                train_part = {'X': X_train[train_ids[i]], 'Y': Y_train[train_ids[i]]}
                # đưa ra giá trị dự đoán và tính hàm loss`
                W = self.fit(train_part['X'], train_part['Y'], LAMBDA)
                Y_predict = self.predict(W, valid_part['X'])
                aver_RSS += self.compute_RSS(valid_part['Y'], Y_predict)
            return aver_RSS / num_folds

        # chọn Lambda có giá trị loss thấp nhất
        def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values):
            for current_LAMBDA in LAMBDA_values:
                aver_RSS = cross_validation(num_folds=5, LAMBDA=current_LAMBDA)
                if aver_RSS < minimum_RSS:
                    best_LAMBDA = current_LAMBDA
                    minimum_RSS = aver_RSS
            return best_LAMBDA, minimum_RSS

        # chạy lần 1 để chọn cận trêm của hàm loss
        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=0,
                                              minimum_RSS=10000 ** 2,
                                              LAMBDA_values=[i * 1.0 / 1000 for i in range(1, 1000, 1)])
        # print('Best LAMBDA: ', best_LAMBDA)
        # print('Minimum RSS: ', minimum_RSS)
        # chạy lần 2 để đưa ra giá trị Lambda tốt nhất
        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=best_LAMBDA,
                                              minimum_RSS=minimum_RSS,
                                              LAMBDA_values=[i * 5.0 / 10000 for i in range(1, 2000, 1)])

        return best_LAMBDA

    def get_the_best_lr(self, X_train, Y_train, LAMBDA):
        """
        Lựa chọn learning rate bằng phương pháp k-fold cross validation
        :param X_train, Y_train: Tập các điểm dữ liệu huấn luyện và nhãn
        :param LAMBDA:
        :return: giá trị Lambda làm cho hàm loss đạt giá trị nhỏ nhất trên
            tập huấn luyện
        """

        def cross_validation(num_folds, lr):
            # chia tập dữ liệu huấn luyện thành k phần
            row_ids = np.array(range(X_train.shape[0]))
            valid_ids = np.split(row_ids[:(len(row_ids) - len(row_ids) % num_folds)], num_folds)
            valid_ids[-1] = np.append(valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds:])
            train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
            aver_RSS = 0
            for i in range(num_folds):
                # với mỗi fold lấy 1 phần làm tập val. k - 1 phần làm tập train
                valid_part = {'X': X_train[valid_ids[i]], 'Y': Y_train[valid_ids[i]]}
                train_part = {'X': X_train[train_ids[i]], 'Y': Y_train[train_ids[i]]}
                # đưa ra giá trị dự đoán và tính hàm loss`
                W = self.fit_gradient_descent(train_part['X'], train_part['Y'], LAMBDA, lr)
                Y_predict = self.predict(W, valid_part['X'])
                aver_RSS += self.compute_RSS(valid_part['Y'], Y_predict)
            return aver_RSS / num_folds

        # chọn Lambda có giá trị loss thấp nhất
        def range_scan(best_lr, minimum_RSS, lr_values):
            for current_lr in lr_values:
                aver_RSS = cross_validation(num_folds=5, lr=current_lr)
                if aver_RSS < minimum_RSS:
                    best_lr = current_lr
                    minimum_RSS = aver_RSS
            return best_lr, minimum_RSS

        # chạy lần 1 để chọn cận trêm của hàm loss
        best_lr, minimum_RSS = range_scan(best_lr=0,
                                          minimum_RSS=10000 ** 2,
                                          lr_values=[i * 1.0 / 1000 for i in range(1, 1000, 1)])
        # print('Best lr: ', best_lr)
        # print('Minimum RSS: ', minimum_RSS)
        # chạy lần 2 để đưa ra giá trị lr tốt nhất
        # best_lr, minimum_RSS = range_scan(best_lr=best_lr,
        #                                   minimum_RSS=minimum_RSS,
        #                                   lr_values=[i * 5.0 / 10000 for i in range(1, 2000, 1)])

        return best_lr


if __name__ == '__main__':
    X, Y = load_data(data_path='datasets/death-rates-data.txt')
    X = normalize_and_add_ones(X)
    # chia tập dữ liệu thành train và test
    X_train, Y_train = X[:50], Y[:50]
    X_test, Y_test = X[50:], Y[50:]

    ridge_regression = RidgeRegression()
    LAMBDA = ridge_regression.get_the_best_LAMBDA(X_train, Y_train)
    print('Best LAMBDA: ', LAMBDA)
    # sử dụng công thức nghiệm
    W_learned = ridge_regression.fit(X_train, Y_train, LAMBDA)
    Y_predicted = ridge_regression.predict(W=W_learned, X_new=X_test)
    print('Loss khi dùng công thức nghiệm', ridge_regression.compute_RSS(Y_new=Y_test, Y_predicted=Y_predicted))

    # sử dụng phương pháp gradient
    lr = ridge_regression.get_the_best_lr(X_train, Y_train, LAMBDA)
    print('Best learning rate: ', lr)
    W_gradient = ridge_regression.fit_gradient_descent(X_train, Y_train, LAMBDA, lr, batch_size=4)
    Y_predicted_gradient = ridge_regression.predict(W=W_gradient, X_new=X_test)

    print('Loss khi dùng gradient', ridge_regression.compute_RSS(Y_new=Y_test, Y_predicted=Y_predicted_gradient))
