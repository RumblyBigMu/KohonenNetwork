import KohNet
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QSpinBox, \
    QGridLayout, QMenuBar, QMessageBox, QLabel, QSizePolicy, QDoubleSpinBox


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QtGui.QIcon('Icon.jpg'))
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Kohonen Network")
        self.setMinimumSize(200, 200)
        self.resize(400, 350)
        self.setMaximumSize(650, 650)
        # Генерация меню
        self.createMenu()
        # Установка центрального виджета
        self.central = CentralWidget(self, self.menu_buttons)
        self.setCentralWidget(self.central)
        self.show()

    def createMenu(self):
        # Настройка меню
        self.menuBar = QMenuBar(self)
        self.setMenuBar(self.menuBar)
        # Меню "Settings"
        menu_settings = self.menuBar.addMenu("Settings")
        ### Построение диаграмм
        plot_action = menu_settings.addAction("Plot Graphs", self.plotGraphs, shortcut='F1')
        plot_action.setDisabled(True)
        ### Запись данных в файл
        write_action = menu_settings.addAction("Write Results to Files", self.writeToFile, shortcut='F2')
        write_action.setDisabled(True)
        # Меню "About"
        menu_about = self.menuBar.addMenu("About")
        menu_about.addAction("About", self.about, shortcut='F3')
        ### Информация о программе
        self.menu_buttons = (plot_action, write_action)

    def about(self):
        text = "Teuvo Kalevi Kohonen (11 July 1934 – 13 December 2021) was a prominent Finnish academic (Dr. Eng.) and researcher. " \
               "He was professor emeritus of the Academy of Finland. " \
               "Prof. Kohonen made many contributions to the field of artificial neural networks, " \
               "including the Learning Vector Quantization algorithm, fundamental theories of " \
               "distributed associative memory and optimal associative mappings, the learning " \
               "subspace method and novel algorithms for symbol processing like redundant hash addressing. " \
               "He published several books and over 300 peer-reviewed papers. " \
               "Kohonen’s most famous contribution is the Self-Organizing Map, also known as the Kohonen " \
               "map or Kohonen artificial neural networks, although Kohonen himself prefers SOM. " \
               "Due to the popularity of the SOM algorithm in many research and in practical applications, " \
               "Kohonen is often considered to be the most cited Finnish scientist."
        QMessageBox.about(self, "About", text)

    def writeToFile(self):
        self.central.Network.KN.writeToFile()
        self.central.Network.KN.writeToExcel()
        self.central.info_label.setText("Data recording completed")
        self.central.info_label.setStyleSheet("color: red")
        # self.central.info_label.setFont(QFont('Arial', 13))

    def plotGraphs(self):
        self.form = PlotWindow(self.central.Network.KN)
        self.form.show()


class CentralWidget(QWidget):
    def __init__(self, parent=None, menu_buttons=None):
        QWidget.__init__(self, parent)
        self.menu_buttons = menu_buttons
        self.initUI()

    def initUI(self):
        self.Network = NetworkWidget(self)

        # Создание спин-боксов
        self.__configureSpins__()
        ### Выбор числа классов для кластеризации
        self.class_spin_label = QLabel(self)
        self.class_spin_label.setText("Please, select the number of classes: ")
        ### Выбор коэффициента обучения
        self.lr_spin_label = QLabel(self)
        self.lr_spin_label.setText("Please, select the network learning rate: ")
        # Кнопка начала обучения
        self.start_training = QPushButton("Start training", self)
        self.start_training.clicked.connect(self.startTraining)
        # Информативный лэйбл
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        # Кнопка сброса интерфейса
        self.reset_button = QPushButton("Reset", self)
        self.reset_button.setDisabled(True)
        self.reset_button.clicked.connect(self.reset)

        # Расположение элементов в сетке
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(20, 20, 20, 20)
        self.grid.setSpacing(15)

        self.grid.addWidget(self.class_spin_label, 0, 0, 1, 2)
        self.grid.addWidget(self.class_spin, 0, 2)

        self.grid.addWidget(self.lr_spin_label, 1, 0, 1, 2)
        self.grid.addWidget(self.lr_spin, 1, 2)

        self.grid.addWidget(self.start_training, 2, 1)
        self.grid.addWidget(self.info_label, 3, 1)

        self.grid.addWidget(self.reset_button, 4, 1)

        self.setLayout(self.grid)

    def __configureSpins__(self):
        # Выбор числа классов для кластеризации
        self.class_spin = QSpinBox(self)
        self.class_spin.setRange(2, 9)
        self.class_spin.setValue(3)
        self.class_spin.setSingleStep(1)
        # Выбор коэффициента обучения
        self.lr_spin = QDoubleSpinBox(self)
        self.lr_spin.setRange(0.05, 1)
        self.lr_spin.setValue(0.3)
        self.lr_spin.setSingleStep(0.05)

    def startTraining(self):
        self.class_spin.setDisabled(True)
        self.lr_spin.setDisabled(True)
        for btn in self.menu_buttons:
            btn.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.Network.KN.K = self.class_spin.value()
        self.Network.KN.Lr = self.lr_spin.value()
        self.Network.KN.learning()
        self.info_label.setText((f"The training took {self.Network.KN.time:0.4f} seconds"))

    def reset(self):
        self.class_spin.setEnabled(True)
        self.lr_spin.setEnabled(True)
        for btn in self.menu_buttons:
            btn.setDisabled(True)
        self.class_spin.setValue(3)
        self.lr_spin.setValue(0.3)
        self.info_label.clear()
        self.info_label.setStyleSheet("color: black")


class NetworkWidget(QWidget):
    def __init__(self, parent=None, classes=3, learningRate=0.3):
        QWidget.__init__(self, parent)
        # Настройки параметров сети
        self.K = classes
        self.Lr = learningRate
        self.KN = KohNet.KohonenNetwork(self.K, self.Lr)

    def learning(self):
        # Обучение сети
        self.KN.learning()

    def saveResults(self):
        # Запись результатов в файлы
        self.KN.writeToExcel()
        self.KN.writeToFile()


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, KN=None):
        # Настройки фигуры
        fig = Figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.KN = KN
        self.plot()

    def plot(self):
        self.figure.clear()
        labels = np.array(
            ["GP", "G", "A", "PTS", "+/-", "PIM", "Shots", "Shots,%", "ATOI", "BLK", "HIT", "FOW",
             "FOL", "FO,%"])
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        labels = np.concatenate((labels, [labels[0]]))
        for i in range(self.KN.K):
            stats = self.KN.WX[i]
            # Настройка графика
            stats = np.concatenate((stats, [stats[0]]))
            ax = self.figure.add_subplot(100 * (self.KN.K // 2) + 30 + (i + 1), projection='polar')
            ax.plot(angles, stats, 'o-', linewidth=2)
            ax.fill(angles, stats, alpha=0.25)
            ax.set_thetagrids(angles * 180 / np.pi, labels)
            ax.set_title("Класс " + str(i + 1))
            ax.grid(True)
        self.figure.suptitle("Лепестковые диаграммы характеристик классов")
        self.figure.tight_layout()
        self.draw()


class PlotWindow(QMainWindow):
    def __init__(self, KN=None):
        super().__init__()
        self.setWindowIcon(QtGui.QIcon('NN.jpg'))
        self.title = "Polar charts"
        self.width = 800
        self.height = 600
        self.KN = KN
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setMinimumSize(self.width, self.height)
        plot = PlotCanvas(self, self.KN)
        self.show()


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())