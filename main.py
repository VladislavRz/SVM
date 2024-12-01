import warnings
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from svm_model import SVM

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (8,6)


class ConfusionMatrix:
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    def __init__(self, class_name):
        self.name = class_name

    def print_precision(self):
        precision = self.TP/(self.TP + self.FP)
        print(f'Precision для класса {self.name} составляет: {precision*100:.2f}%')

    def print_recall(self):
        recall = self.TP/(self.TP + self.FN)
        print(f'Recall для класса {self.name} составляет: {recall*100:.2f}%')

    def print_fm(self):
        precision = self.TP / (self.TP + self.FP)
        recall = self.TP / (self.TP + self.FN)
        print(f'F-мера для класса {self.name} составляет: {2*precision*recall*100/(precision+recall):.2f}%')


def newline(p1, p2, color=None):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax], color=color)
    ax.add_line(l)
    return l


if __name__ == '__main__':

    # Загрузка и подготовка данных
    iris = load_iris()
    X = iris.data
    Y = iris.target
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    Y = (Y > 0).astype(int) * 2 - 1
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.67, random_state=2)

    # Тренировка модели
    svm = SVM(step=0.005, epochs=150)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)

    accurancy = 0
    stat = [ConfusionMatrix(y) for y in set(y_test.astype(int))]

    # Расчет характеристик
    for i, j in zip(y_test.astype(int), y_pred.astype(int)):
        actual = i
        prediction = j

        for matrix in stat:
            if (matrix.name == actual) and (matrix.name == prediction):
                matrix.TP += 1
            elif (matrix.name == actual) and (matrix.name != prediction):
                matrix.FN += 1
            elif (matrix.name != actual) and (matrix.name == prediction):
                matrix.FP += 1
            elif (matrix.name != actual) and (matrix.name != prediction):
                matrix.TN += 1

        if actual == prediction:
            accurancy += 1

    for matrix in stat:
        matrix.print_precision()
        matrix.print_recall()
        matrix.print_fm()
        print()

    print(f"\nТочность предсказаний модели: {(accurancy/y_test.size)*100:.2f}%")

    # Вывод тренировочных данных
    d = {-1: 'green', 1: 'red'}
    plt.scatter(x_train[:, 0], x_train[:, 1], c=[d[y] for y in y_train])
    newline([0, -svm.weights[2] / svm.weights[1]], [-svm.weights[2] / svm.weights[0], 0], 'blue')
    plt.show()

    # предсказываем после обучения
    d1 = {-1: 'lime', 1: 'm'}
    plt.scatter(x_test[:, 0], x_test[:, 1], c=[d1[y] for y in y_pred])
    newline([0, -svm.weights[2] / svm.weights[1]], [-svm.weights[2] / svm.weights[0], 0], 'blue')
    plt.show()

