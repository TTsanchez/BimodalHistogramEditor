import sys
import traceback
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QSlider, QSpinBox, QGroupBox, QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class HistogramCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def plot_histogram(self, image, title="Histogram"):
        self.ax.clear()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        self.ax.plot(hist, color='black')
        self.ax.set_title(title)
        self.draw()


class ImageViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(512, 512)
        self.original_pixmap = None
        self.processed_pixmap = None
        self.show_original = True
        self.pixel_scale = 1  # Режим 1:1
        self.last_pos = None
        self.setMouseTracking(True)
        self.show_original = True  # По умолчанию показываем оригинал

    def set_images(self, original, processed, show_result=False):
        self.original_pixmap = original
        self.processed_pixmap = processed
        if show_result and processed:
            self.show_original = False
        self.update()

    def set_scale(self, scale):
        self.pixel_scale = scale
        self.update()

    def toggle_image(self):
        self.show_original = not self.show_original
        self.update()

    def mouseMoveEvent(self, event):
        self.last_pos = event.pos()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        current = self.original_pixmap if self.show_original else self.processed_pixmap
        if current:
            if self.pixel_scale == 1:
                painter.drawPixmap(0, 0, current)
            else:
                scaled = current.scaled(
                    current.size() * self.pixel_scale,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                painter.drawPixmap(0, 0, scaled)

        # Отображение информации о пикселе
        if self.last_pos and current:
            x = self.last_pos.x() // self.pixel_scale
            y = self.last_pos.y() // self.pixel_scale

            if 0 <= x < current.width() and 0 <= y < current.height():
                img = current.toImage()
                color = img.pixelColor(x, y)

                # Рисуем прямоугольник с информацией
                painter.setPen(QPen(Qt.black, 1))
                painter.setBrush(QColor(255, 255, 255, 200))
                painter.drawRect(10, 10, 230, 30)

                # Текст информации
                painter.setPen(Qt.black)
                font = QFont()
                font.setPointSize(10)
                painter.setFont(font)

                text = f"X: {x}, Y: {y} \n"
                if color:
                    text += f"R: {color.red()}, G: {color.green()}, B: {color.blue()}"
                painter.drawText(20, 30, text)


class ImageProcessor:
    @staticmethod
    def equalize_histogram(image):
        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return cv2.equalizeHist(image)

    @staticmethod
    def create_bimodal_histogram(peaks, sigma=30):
        x = np.arange(256)
        hist = np.zeros(256)
        for peak in peaks:
            hist += np.exp(-(x - peak) ** 2 / (2 * sigma ** 2))
        return hist / hist.sum()

    @staticmethod
    def match_histogram(image, target_hist):
        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = ImageProcessor._match_channel(ycrcb[:, :, 0], target_hist)
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return ImageProcessor._match_channel(image, target_hist)

    @staticmethod
    def _match_channel(channel, target_hist):
        # Вычисляем CDF для исходного изображения
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        source_cdf = hist.cumsum()

        # Вычисляем CDF для целевой гистограммы
        target_cdf = target_hist.cumsum()

        # Создаем LUT
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            lut[i] = np.argmin(np.abs(source_cdf[i] - target_cdf))

        return cv2.LUT(channel, lut)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Histogram Processor")
        self.setGeometry(100, 100, 1200, 800)

        self.original_image = None
        self.processed_image = None
        self.bimodal_peaks = (85, 170)

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Левая панель - изображения
        left_panel = QVBoxLayout()

        # Image viewer
        self.image_viewer = ImageViewer()
        left_panel.addWidget(self.image_viewer)

        # Кнопки управления
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.save_btn = QPushButton("Save Image")
        self.equalize_btn = QPushButton("Equalize Histogram")
        self.bimodal_btn = QPushButton("Match Bimodal")
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.equalize_btn)
        btn_layout.addWidget(self.bimodal_btn)
        left_panel.addLayout(btn_layout)

        # Правая панель - гистограммы и настройки
        right_panel = QVBoxLayout()

        # Гистограммы
        self.original_hist = HistogramCanvas()
        self.processed_hist = HistogramCanvas()
        right_panel.addWidget(QLabel("Original Histogram"))
        right_panel.addWidget(self.original_hist)
        right_panel.addWidget(QLabel("Processed Histogram"))
        right_panel.addWidget(self.processed_hist)

        # Настройки бимодальной гистограммы
        settings_group = QGroupBox("Bimodal Histogram Settings")
        settings_layout = QVBoxLayout()

        # Пик 1
        peak1_layout = QHBoxLayout()
        peak1_layout.addWidget(QLabel("Peak 1:"))
        self.peak1_slider = QSlider(Qt.Horizontal)
        self.peak1_slider.setRange(0, 255)
        self.peak1_slider.setValue(85)
        peak1_layout.addWidget(self.peak1_slider)
        self.peak1_value = QLabel("85")
        peak1_layout.addWidget(self.peak1_value)
        settings_layout.addLayout(peak1_layout)

        # Пик 2
        peak2_layout = QHBoxLayout()
        peak2_layout.addWidget(QLabel("Peak 2:"))
        self.peak2_slider = QSlider(Qt.Horizontal)
        self.peak2_slider.setRange(0, 255)
        self.peak2_slider.setValue(170)
        peak2_layout.addWidget(self.peak2_slider)
        self.peak2_value = QLabel("170")
        peak2_layout.addWidget(self.peak2_value)
        settings_layout.addLayout(peak2_layout)

        settings_group.setLayout(settings_layout)
        right_panel.addWidget(settings_group)

        main_layout.addLayout(left_panel, 70)
        main_layout.addLayout(right_panel, 30)
        self.setCentralWidget(main_widget)

    def setup_connections(self):
        self.load_btn.clicked.connect(self.load_image)
        self.save_btn.clicked.connect(self.save_image)
        self.equalize_btn.clicked.connect(self.equalize_image)
        self.bimodal_btn.clicked.connect(self.match_bimodal)
        self.peak1_slider.valueChanged.connect(self.update_peaks)
        self.peak2_slider.valueChanged.connect(self.update_peaks)
        self.image_viewer.mousePressEvent = lambda e: self.image_viewer.toggle_image()

    def update_peaks(self):
        self.bimodal_peaks = (self.peak1_slider.value(), self.peak2_slider.value())
        self.peak1_value.setText(str(self.peak1_slider.value()))
        self.peak2_value.setText(str(self.peak2_slider.value()))

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.bmp *.png *.jpg *.jpeg)")

        if file_name:
            self.original_image = cv2.imread(file_name)
            if self.original_image is not None:
                self.processed_image = None
                # При загрузке показываем оригинал
                self.update_display(show_result=False)

    def save_image(self):
        if self.processed_image is not None:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "",
                "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)")

            if file_name:
                cv2.imwrite(file_name, self.processed_image)

    def equalize_image(self, equalized=None):
        """Эквализация с обработкой ошибок"""
        try:
            if self.original_image is None:
                QMessageBox.warning(self, "Warning", "No image loaded")
                return

            # Создаем копию для обработки
            img = self.original_image.copy()

            # Эквализация в зависимости от типа изображения
            if len(img.shape) == 3:  # Цветное
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                y_channel = ycrcb[:, :, 0]

                # Ручная эквализация для канала Y
                hist = np.zeros(256, dtype=np.float32)
                for i in range(y_channel.shape[0]):
                    for j in range(y_channel.shape[1]):
                        hist[y_channel[i, j]] += 1

                hist /= (y_channel.shape[0] * y_channel.shape[1])
                cdf = hist.cumsum()
                cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
                cdf = cdf.astype(np.uint8)

                equalized = np.zeros_like(y_channel)
                for i in range(y_channel.shape[0]):
                    for j in range(y_channel.shape[1]):
                        equalized[i, j] = cdf[y_channel[i, j]]

                ycrcb[:, :, 0] = equalized
                self.processed_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:  # Grayscale
                # Аналогичная обработка для серых изображений
                hist = np.zeros(256, dtype=np.float32)
                # ... (реализация аналогична цветному варианту)
                self.processed_image = equalized

            self.update_display(show_result=True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Equalization failed: {str(e)}")
            print(f"Error details: {traceback.format_exc()}")

    def match_bimodal(self):
        """Бимодальное преобразование с обработкой ошибок"""
        try:
            if self.original_image is None:
                QMessageBox.warning(self, "Warning", "No image loaded")
                return

            # Создаем копию для обработки
            img = self.original_image.copy()

            # Создаем целевую гистограмму
            target_hist = np.zeros(256)
            for peak in self.bimodal_peaks:
                for i in range(256):
                    target_hist[i] += np.exp(-(i - peak) ** 2 / (2 * 30 ** 2))
            target_hist /= target_hist.sum()

            # Применяем преобразование
            if len(img.shape) == 3:  # Цветное
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                y_channel = ycrcb[:, :, 0]

                # Расчет гистограммы исходного изображения
                hist = np.zeros(256, dtype=np.float32)
                for i in range(y_channel.shape[0]):
                    for j in range(y_channel.shape[1]):
                        hist[y_channel[i, j]] += 1

                hist /= (y_channel.shape[0] * y_channel.shape[1])
                source_cdf = hist.cumsum()

                # Расчет CDF целевой гистограммы
                target_cdf = target_hist.cumsum()

                # Создание LUT
                lut = np.zeros(256, dtype=np.uint8)
                for i in range(256):
                    lut[i] = np.argmin(np.abs(source_cdf[i] - target_cdf))

                # Применение преобразования
                matched = np.zeros_like(y_channel)
                for i in range(y_channel.shape[0]):
                    for j in range(y_channel.shape[1]):
                        matched[i, j] = lut[y_channel[i, j]]

                ycrcb[:, :, 0] = matched
                self.processed_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:  # Grayscale
                # Аналогичная обработка для серых изображений
                pass

            self.update_display(show_result=True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Bimodal transform failed: {str(e)}")
            print(f"Error details: {traceback.format_exc()}")

    def update_display(self, show_result=False):
        """Добавляем параметр show_result"""
        if self.original_image is not None:
            original_qt = self.cv2_to_qimage(self.original_image)
            original_pix = QPixmap.fromImage(original_qt)

            processed_pix = None
            if self.processed_image is not None:
                processed_qt = self.cv2_to_qimage(self.processed_image)
                processed_pix = QPixmap.fromImage(processed_qt)

            # Передаем параметр show_result
            self.image_viewer.set_images(original_pix, processed_pix, show_result)

            self.original_hist.plot_histogram(self.original_image, "Original Histogram")
            if self.processed_image is not None:
                self.processed_hist.plot_histogram(self.processed_image, "Processed Histogram")

    def cv2_to_qimage(self, cv_img):
        if cv_img is None:
            return None

        if len(cv_img.shape) == 2:
            return QImage(cv_img.data, cv_img.shape[1], cv_img.shape[0],
                          cv_img.shape[1], QImage.Format_Grayscale8)
        elif len(cv_img.shape) == 3:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            return QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                          rgb.shape[1] * 3, QImage.Format_RGB888)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())