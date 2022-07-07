import numpy as np
import pandas as pd
import Dataset as Dt
import matplotlib.pyplot as plt


class KohonenNetwork():
    def __init__(self, classNum):
        self.K = classNum
        self.X = Dt.NHL  # Массив исходных данных
        self.M = len(self.X)  # Количество исходных данных
        self.N = len(self.X[0]) - 2  # Размерность векторов
        self.W = np.random.rand(self.K, self.N)  # Инциализация массива весов случайными значениями в диапазоне [0, 1)
        self.As, self.Bs, self.X = KohonenNetwork.__normalization__(self.X)  # Нормировка исходных данных
        self.la = 0.3  # Коэффициент обучения
        self.dla = 0.05  # Уменьшение коэффициента обучения

        self.learning()

    def learning(self):
        while self.la >= 0:
            for k in range(10):  # Повторение процесса обучения 10 раз
                for x in self.X:
                    wm = KohonenNetwork.findNear(self.W, x[2:])[0]
                    for i in range(len(wm)):
                        wm[i] = wm[i] + self.la * (x[i + 2] - wm[i])  # Корректировка весов
            self.la = self.la - self.dla  # Уменьшение коэффициента обучения
        # Массив денормированных весов
        self.WX = list()
        # Денормировка весов
        for i in range(self.K):
            self.WX.append(list())
            for j in range(self.N):
                self.WX[i].append((self.W[i][j] - self.Bs[j]) / self.As[j])
        self.Data, self.DS = KohonenNetwork.__clusterization__(self.W, self.X)

    # Расстояние между векторами
    @staticmethod
    def rho(a, b):
        sum = 0
        for i in range(len(a)):
            sum = sum + (a[i] - b[i]) * (a[i] - b[i])
        sum = np.sqrt(sum)
        return sum

    # Поиск ближайшего вектора
    @staticmethod
    def findNear(W, x):
        wm = W[0]
        r = KohonenNetwork.rho(wm, x)
        i = 0
        i_n = i
        for w in W:
            if KohonenNetwork.rho(w, x) < r:
                r = KohonenNetwork.rho(w, x)
                wm = w
                i_n = i
            i = i + 1
        return wm, i_n

    # Нормализация данных
    @staticmethod
    def __normalization__(X):
        # Коэффиценты нормировки
        As = list()
        Bs = list()
        for i in range(2, len(X[0])):
            M = X[0][i]
            m = X[0][i]
            for j in range(len(X)):
                if abs(X[j][i]) > M:
                    M = abs(X[j][i])
                else:
                    if abs(X[j][i]) < m:
                        m = abs(X[j][i])
            # Коэффициенты нормировки
            a = 1 / (M - m)
            b = -m / (M - m)
            # Сохранение коэффициентов
            As.append(a)
            Bs.append(b)
            # Нормирование
            for j in range(len(X)):
                X[j][i] = a * X[j][i] + b
        return As, Bs, X

    # Кластеризация данных
    @staticmethod
    def __clusterization__(W, X):
        # Создание классов
        Data = list()
        for i in range(len(W)):
            Data.append(list())
        # Соотношение исходных данных с полученными классами
        DS = list()
        i = 0
        for x in X:
            i_n = KohonenNetwork.findNear(W, x[2:])[1]
            Data[i_n].append(x[2:])
            DS.append([i_n, x[0], x[1]])
            i = i + 1
        return Data, DS

    # Печать количества элементов класса
    def printClasses(self):
        k = 1
        class_n = list()
        for d in self.Data:
            print("Класс " + str(k) + " состоит из " + str(len(d)) + " элементов")
            class_n.append(len(d))
            k += 1
        class_n.sort()
        print(class_n)

    # Запись данных о характерных значениях каждого класса в таблицу
    def writeToExcel(self):
        Stats = pd.DataFrame(self.WX, columns=[Dt.Stats[2:]])
        Stats.to_excel('Data\Statistics.xlsx')

    # Запись данных о классах в файлы
    def writeToFile(self):
        f = list()
        for i in range(self.K):
            f.append(open("Data/" + str(i + 1) + '_NHL.txt', "w"))
        for ds in self.DS:
            f[ds[0]].write(ds[1])
            f[ds[0]].write('\t')
            f[ds[0]].write(ds[2])
            f[ds[0]].write('\n')
        for i in range(self.K):
            f[i].close()

    # Отрисовка лепестковых диаграмм
    def plot(self):
        labels = np.array(
            ["GP", "G", "A", "PTS", "+/-", "PIM", "Shots", "Shots,%", "ATOI", "BLK", "HIT", "FOW",
             "FOL", "FO,%"])
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        labels = np.concatenate((labels, [labels[0]]))
        # Создание фигуры
        fig = plt.figure()
        for i in range(self.K):
            stats = self.WX[i]
            # Настрока графика
            stats = np.concatenate((stats, [stats[0]]))
            ax = plt.subplot(100 * (self.K // 2) + 30 + (i + 1), projection='polar')
            ax.plot(angles, stats, 'o-', linewidth=2)
            ax.fill(angles, stats, alpha=0.25)
            ax.set_thetagrids(angles * 180 / np.pi, labels)
            ax.set_title("Класс " + str(i + 1))
            ax.grid(True)
        plt.suptitle("Лепестковые диаграммы характеристик классов")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    print("Пожалуйста, введите количество классов: ", end=' ')
    K = int(input())  # Количество классов
    cl = KohonenNetwork(K)
    cl.printClasses()
    cl.writeToExcel()
    cl.writeToFile()
    cl.plot()