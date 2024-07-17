import sys
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QWidget, QVBoxLayout, QLabel , QMessageBox, QGraphicsScene, QGraphicsView 
from PyQt5.QtGui import QPainter, QPen, QPixmap , QImage
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QPoint
import sys
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchsummary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchsummary import summary
from torchvision.models import resnet50

app = QtWidgets.QApplication(sys.argv) 
global pic1
filter_type = 0
window = QtWidgets.QMainWindow()
window.setWindowTitle("HW2")
window.setGeometry(300, 100, 1000, 800) 
loadimage = QtWidgets.QPushButton("Load Image ", window)
loadimage.move(40, 250)

# Create Q1
groupBox1= QtWidgets.QGroupBox("1.Hough Circle Transform ", window)
groupBox1.setFixedSize(250 , 230)
groupBox1.move(180, 30)
pushButton1_1 = QtWidgets.QPushButton("1.1 Draw Contour", window)
pushButton1_1.resize(180, 40)
pushButton1_1.move(220, 60)
pushButton1_2 = QtWidgets.QPushButton("1.2 Count Rings", window)
pushButton1_2.resize(180, 40)
pushButton1_2.move(220, 125)
label1_1 = QtWidgets.QLabel("There are           coins in the image", window)
label1_1.resize(180, 40)
label1_1.move(220, 190)
label1_2 = QtWidgets.QLabel("", window)
label1_2.resize(180, 40)
label1_2.move(275, 190)

#Create Q2
groupBox2= QtWidgets.QGroupBox("2.Histogram Equalization", window)
groupBox2.setFixedSize(250 , 100)
groupBox2.move(180, 280)
pushButton2_1 = QtWidgets.QPushButton("2.1 Histogram Equalization", window)
pushButton2_1.resize(180, 40)
pushButton2_1.move(220, 310)

#Create Q3
groupBox3= QtWidgets.QGroupBox("3.Morphology Operation ", window)
groupBox3.setFixedSize(250 , 130)
groupBox3.move(180, 400)
pushButton3_1 = QtWidgets.QPushButton("3.1 Closing", window)
pushButton3_1.resize(180, 40)
pushButton3_1.move(220, 420)
pushButton3_2 = QtWidgets.QPushButton("3.2 Opening", window)
pushButton3_2.resize(180, 40)
pushButton3_2.move(220, 470)

#create Q4
groupBox4 = QtWidgets.QGroupBox("4.MNIST Classifier Using VGG19", window)
groupBox4.setFixedSize(530 , 350)
groupBox4.move(450, 30)
pushButton4_1 = QtWidgets.QPushButton("4.1 Show Model Structure", window)
pushButton4_1.resize(130, 30)
pushButton4_1.move(470 , 70)
pushButton4_2 = QtWidgets.QPushButton("4.2 Show Accuracy and Loss", window)
pushButton4_2.resize(150, 40)
pushButton4_2.move(460, 120)
pushButton4_3 = QtWidgets.QPushButton("4.3 Predict", window)
pushButton4_3.resize(130, 30)
pushButton4_3.move(470, 170)
pushButton4_4 = QtWidgets.QPushButton("4.4 Reset", window)
pushButton4_4.resize(130, 30)
pushButton4_4.move(470, 220)
label4 = QtWidgets.QLabel("", window)
label4.resize(180, 40)
label4.move(470, 260)

#Create Q5
groupBox5= QtWidgets.QGroupBox("5.ResNet50", window)
groupBox5.setFixedSize(530 , 350)
groupBox5.move(450, 400)
pushButton5 = QtWidgets.QPushButton("Load Image", window)
pushButton5.resize(180, 30)
pushButton5.move(470, 450)
pushButton5_1 = QtWidgets.QPushButton("5.1 Show Images", window)
pushButton5_1.resize(180, 30)
pushButton5_1.move(470, 500)
pushButton5_2 = QtWidgets.QPushButton("5.2 Show Model Structure", window)
pushButton5_2.resize(180, 30)
pushButton5_2.move(470, 550)
pushButton5_3 = QtWidgets.QPushButton("5.3 Show Comparasion", window)
pushButton5_3.resize(180, 30)
pushButton5_3.move(470, 600)
pushButton5_4 = QtWidgets.QPushButton("5.4 Inference", window)
pushButton5_4.resize(180, 30)
pushButton5_4.move(470, 650)
label5 = QtWidgets.QLabel("", window)
label5.resize(224,224)
label5.move(660, 460)
label5_4 = QtWidgets.QLabel("Prediction : ", window)
label5_4.resize(180, 40)
label5_4.move(660, 700)

class GraffitiBoard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.brushColor = Qt.white
        self.brushSize = 2
        self.lastPoint = QPoint()
        self.currentPoint = QPoint()
        self.initUI()

    def initUI(self):
        self.canvas = QPixmap(400, 300)
        self.canvas.fill(Qt.black)
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 400, 300)
        self.label.setPixmap(self.canvas)
        mainLayout = QVBoxLayout(self)
        mainLayout.addWidget(self.label)
        pushButton4_4.clicked.connect(self.clearCanvas)
        self.setGeometry(600, 60, 400, 300)

    def paintEvent(self, event):
        painter = QPainter(self.canvas)
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self.drawing:
            local_pos = self.label.mapFromGlobal(self.mapToGlobal(self.lastPoint))
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(local_pos, self.currentPoint)
            self.lastPoint = self.currentPoint
        self.label.setPixmap(self.canvas)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = self.currentPoint = self.label.mapFromGlobal(event.globalPos())
            #self.lastPoint = self.label.mapFromGlobal(event.globalPos())
            self.drawing = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            self.currentPoint = self.label.mapFromGlobal(event.globalPos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def saveCanvasToFile(self , filename):
        self.canvas.save(filename)

    def clearCanvas(self):
        self.canvas.fill(Qt.black)
        self.label.setPixmap(self.canvas)   
graffiti_board = GraffitiBoard(window)

class NumberPredictionApp(QMainWindow):
    def __init__(self, model_path):
        super().__init__()
        self.model = torch.load(model_path)
        self.model.eval()
        self.initUI()
        self.img = Image.new("L", (280, 280), color=0)
        self.draw = QPainter(self.img)

    def initUI(self):
        self.setWindowTitle("Number Prediction App")
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setSceneRect(0, 0, 280, 280)
        pushButton4_3.clicked.connect(self.predict_number)
        pushButton4_4.clicked.connect(self.reset_canvas)
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.reset_button)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.draw_line_to(event.pos())

    def draw_line_to(self, end_point):
        self.draw.setPen(QPen(Qt.white, 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        self.draw.drawLine(self.last_point, end_point)
        self.last_point = end_point
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(self.get_qimage()))

    def get_qimage(self):
        image = QImage(self.img.tobytes(), self.img.width, self.img.height, QImage.Format_Grayscale8)
        return image

    def predict_number(self):
        input_data = np.array(self.img) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        input_tensor = torch.from_numpy(input_data).float()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(output).item()
        QMessageBox.information(self, "Prediction", f"The predicted class is: {predicted_class}")
        plt.bar(range(10), probabilities.squeeze().numpy())
        plt.title("Probability Distribution")
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.show()

    def reset_canvas(self):
        self.draw.end()
        self.img.paste(0, (0, 0, 280, 280))
        self.draw = QPainter(self.img)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(self.get_qimage()))

def loadimage_clicked():
    global pic1
    pic1 = open_image_using_dialog()
    cv2.imshow("Image", pic1)
loadimage.clicked.connect(loadimage_clicked)

def open_image_using_dialog():
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    image_path, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*)", options=options)
    print(image_path)
    if image_path:
        image = cv2.imread(image_path)
    else:
        image = None
    return image

detected_circles = 0
def pushButton1_1_clicked():
    global pic1 , detected_circles
    gray = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
    blurred, 
    cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=50
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        processed_image = pic1.copy()
        circle_centers_image = np.zeros_like(pic1)
    detected_circles = 0
    
    for i in circles[0, :]:
        cv2.circle(processed_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(circle_centers_image, (i[0], i[1]), 5, (255, 255, 255), -1)
        detected_circles += 1
    print(f"Number of detected circles: {detected_circles}")
    plt.figure(figsize=(10, 6))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title('Processed Image ')
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(circle_centers_image, cv2.COLOR_BGR2RGB))
    plt.title('Circle Centers Detected')
    plt.show()
pushButton1_1.clicked.connect(pushButton1_1_clicked)
    
def pushButton1_2_clicked():
    global pic1, detected_circles
    label1_2.setText(str(detected_circles))
pushButton1_2.clicked.connect(pushButton1_2_clicked)

def pushButton2_1_clicked():
    global pic1
    gray = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray)
    hist, bins = np.histogram(gray.flatten(), bins=256, range=[0, 256])
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)
    lookup_table = np.round(cdf * 255).astype('uint8')
    equalized_image_manual = lookup_table[gray]
    equalized_hist, _ = np.histogram(equalized_image_manual.flatten(), 256, [0, 256])
    plt.figure(figsize=(10,6))
    plt.subplot(2, 3, 1), plt.imshow(pic1, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 2), plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized with OpenCV'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 3), plt.imshow(equalized_image_manual, cmap='gray')
    plt.title('Equalized Manually'), plt.xticks([]), plt.yticks([])
    x = [0, 50, 100, 150, 200, 250]      
    h = [0, 5000, 10000, 15000, 20000, 25000]   
    plt.subplot(2, 3, 4)
    plt.hist(gray.flatten(), 256, [0, 256], color='b')
    plt.xlabel('gray scale', fontsize="10") 
    plt.ylabel('frequency', fontsize="10") 
    plt.title('Histogram of Original', fontsize="10")

    plt.subplot(2, 3, 5)
    plt.hist(equalized_image.flatten(), 256, [0, 256], color='blue')
    plt.xlabel('gray scale', fontsize="10")
    plt.ylabel('frequency', fontsize="10") 
    plt.title('Histogram of Equalized (OpenCV)', fontsize="10")

    plt.subplot(2, 3, 6)
    plt.hist(equalized_image_manual.flatten(), 256, [0, 256], color='blue')
    plt.xlabel('gray scale', fontsize="10")
    plt.ylabel('frequency', fontsize="10") 
    plt.title('Histogram of Equalized (Manual)', fontsize="10")

    plt.bar(x,h)
    plt.tight_layout()
    
    plt.show()
pushButton2_1.clicked.connect(pushButton2_1_clicked)

def pushButton3_1_clicked():
    global pic1 
    gray = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    pad_size = 1
    padded_image = np.pad(bw, pad_size, mode='constant', constant_values=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_image = np.zeros_like(padded_image)
    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            dilated_image[i, j] = np.max(padded_image[i-1:i+2, j-1:j+2] * kernel)
    eroded_image = np.zeros_like(dilated_image)
    for i in range(1, dilated_image.shape[0] - 1):
        for j in range(1, dilated_image.shape[1] - 1):
            eroded_image[i, j] = np.min(dilated_image[i-1:i+2, j-1:j+2] * kernel)

    plt.imshow(eroded_image, cmap='gray')
    plt.title('Closed Image')
    plt.show()
pushButton3_1.clicked.connect(pushButton3_1_clicked)

def pushButton3_2_clicked():
    global pic1 
    gray = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    pad_size = 1
    padded_image = np.pad(bw, pad_size, mode='constant', constant_values=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded_image = np.zeros_like(padded_image)
    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            eroded_image[i, j] = np.min(padded_image[i-1:i+2, j-1:j+2] * kernel)

    dilated_image = np.zeros_like(eroded_image)
    for i in range(1, eroded_image.shape[0] - 1):
        for j in range(1, eroded_image.shape[1] - 1):
            dilated_image[i, j] = np.max(eroded_image[i-1:i+2, j-1:j+2] * kernel)

    plt.imshow(dilated_image, cmap='gray')
    plt.title('Opened Image')
    plt.show()
pushButton3_2.clicked.connect(pushButton3_2_clicked)

def pushButton4_1_clicked():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.vgg19_bn(num_classes=10).to(device)
    torchsummary.summary(model, (3, 32, 32))
pushButton4_1.clicked.connect(pushButton4_1_clicked)

def pushButton4_2_clicked():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    relative_path = 'training_validation_metrics.png'
    absolute_path = os.path.join(script_directory, relative_path)
    image = cv2.imread(absolute_path)                               
    cv2.imshow("4-2", image)   
pushButton4_2.clicked.connect(pushButton4_2_clicked)

def predict_number():
    model = torchvision.models.vgg19_bn(num_classes = 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    graffiti_board.saveCanvasToFile("write.png")
    pic2 = Image.open("write.png")
    pic2 = pic2.convert("L")
    
    model_path = "./best_model.pth"  
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    model.eval()
    class_names = ['0','1','2','3','4','5','6','7','8','9',]
    transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    ])

    pic2 = transform(pic2)
    pic2 = pic2.unsqueeze(0)
    pic2 = pic2.to(device)

        # Perform inference
    with torch.no_grad():
        pred = model(pic2)
        pred = torch.softmax(pred, dim=1)
        pred_idx = pred.argmax(dim=1).item()

        # Display the predicted class label

    #predicted_class = probabilities.argmax(dim=1).item()
    label4.setText("{}".format(class_names[pred_idx]))
    pred = pred.squeeze().cpu().numpy()


    # Display the probability distribution in a histogram
    plt.figure(figsize=(10 , 10))
    plt.bar(range(10), pred)
    plt.title("Probability Distribution")
    plt.xticks(range(10))
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.show()
pushButton4_3.clicked.connect(predict_number)

class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

script_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = 'inference_dataset'
inference_data_path = os.path.join(script_directory, relative_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

inference_dataset = InferenceDataset(root_dir=inference_data_path, transform=transform)


inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=True)

classes = inference_dataset.dataset.classes

def pushButton5_1_clicked():
    plt.figure(figsize=(10, 5))

    for i, class_name in enumerate(classes):

        class_images = [img for img, label in inference_dataloader if label == classes.index(class_name)]
        class_image = class_images[0]
        class_image_np = class_image.squeeze().numpy().transpose((1, 2, 0))

        plt.subplot(1, len(classes), i+1)
        plt.imshow(class_image_np)
        plt.title(class_name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
pushButton5_1.clicked.connect(pushButton5_1_clicked)

def pushButton5_2_clicked():
    try:
        resnet50 = models.resnet50(pretrained=False)
        resnet50.load_state_dict(torch.load('resnet50_weights.pth'))
        print("Using pre-trained weights from local file.")
    except:
        resnet50 = models.resnet50(pretrained=True)
        torch.save(resnet50.state_dict(), 'resnet50_weights.pth')
        print("Downloaded and saved pre-trained weights.")
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet50 = resnet50.to(device)
    summary(resnet50, (3, 224, 224))  
pushButton5_2.clicked.connect(pushButton5_2_clicked)    

def pushButton5_3_clicked():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    relative_path = 'accuracy_comparison.png'
    absolute_path = os.path.join(script_directory, relative_path)
    image = cv2.imread(absolute_path)           
    cv2.imshow("5-3", image)   
pushButton5_3.clicked.connect(pushButton5_3_clicked)

class MyResNetModel(nn.Module):
    def __init__(self):
        super(MyResNetModel, self).__init__()
        self.resnet = resnet50(pretrained=True)

    def forward(self, x):
        x = self.resnet(x)
        return x
    
def load_model(model_path):
    model = MyResNetModel() 
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, image_path, threshold=0.5):
    img = Image.fromarray(image_path)
    transform_with_erasing = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing()
])
    img = transform_with_erasing(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = model(img)

    prediction = F.softmax(output, dim=1)
    class_label = "Cat" if prediction[0, 0] > threshold else "Dog"

    return class_label

def inference():
    model = load_model("./trained_model_with_erasing.pth")
    predicted_class = predict(model, pic3, threshold=0.5)
    label5_4.setText("Prediction : {}".format(predicted_class))
pushButton5_4.clicked.connect(inference)

def load_image():
    global pic3
    file_dialog = QFileDialog()
    file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
    file_dialog.setViewMode(QFileDialog.Detail)
    file_path, _ = file_dialog.getOpenFileName()

    if file_path:
        pic3 = cv2.imread(file_path)
        pic3 = cv2.resize(pic3, (224, 224))
        pic3 = cv2.cvtColor(pic3, cv2.COLOR_BGR2RGB)
        height, width, channel = pic3.shape
        bytes_per_line = 3 * width
        q_img = QPixmap.fromImage(QImage(pic3.data, width, height, bytes_per_line, QImage.Format_RGB888))
        label5.setPixmap(q_img)
pushButton5.clicked.connect(load_image)

window.show()

sys.exit(app.exec_())
