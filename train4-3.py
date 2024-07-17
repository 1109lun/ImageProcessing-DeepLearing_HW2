import sys
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel , QMessageBox, QGraphicsScene, QGraphicsView 
from PyQt5.QtGui import QPainter, QColor, QPen, QPixmap , QImage
from PyQt5 import QtWidgets , uic
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
from torchvision import transforms, datasets
from torch.utils import data as data_
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.image as img

model = torchvision.models.vgg19_bn(num_classes = 10)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=3)
])
val_transform = transforms.Compose([transforms.ToTensor() , transforms.Grayscale(num_output_channels=3)])

train_set = datasets.MNIST('data', train=True, download=True, transform=transform)
val_set = datasets.MNIST('data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)

loss_fn = nn.CrossEntropyLoss()

best_val_acc , best_epoch = 0 , 0

num_epoch = 30
avg_train_losses = []
avg_train_accs = []
avg_val_losses = []
avg_val_accs = []

for epoch in range(num_epoch):
        # Set the model in training mode
    model.train()
    train_loss_history = []
    train_acc_history = []

    for x , y in train_loader:
        x , y = x.cuda() , y.cuda()
        y_one_hot = nn.functional.one_hot(y , num_classes=10).float()
        y_pred = model(x)

        loss = loss_fn(y_pred , y_one_hot)
        loss.backward()

        optimizer.step() 
        optimizer.zero_grad()

        acc = (y_pred.argmax(dim = 1) == y).float().mean()
        train_loss_history.append(loss.item())
        train_acc_history.append(acc.item())

    avg_train_loss = sum(train_loss_history) / len(train_loss_history)
    avg_train_accuracy = sum(train_acc_history) / len(train_acc_history)
    avg_train_losses.append(avg_train_loss)
    avg_train_accs.append(avg_train_accuracy)

    model.eval()
    val_loss_history = []
    val_acc_history = []

    for x , y in test_loader:
        x , y = x.cuda() , y.cuda()
        y_one_hot = nn.functional.one_hot(y , num_classes=10).float()
        with  torch.no_grad():
            y_pred = model(x)
            loss = loss_fn(y_pred , y_one_hot)
            acc = (y_pred.argmax(dim = 1) == y).float().mean()
        val_loss_history.append(loss.item())
        val_acc_history.append(acc.item())
        
    avg_val_loss = sum(val_loss_history) / len(val_loss_history)
    avg_val_acc = sum(val_acc_history) / len(val_acc_history)
    avg_val_losses.append(avg_val_loss)
    avg_val_accs.append(avg_val_acc)

    if avg_val_acc >= best_val_acc :
        print('Best model saved at epoch {} , acc : {:.4f}'.format(epoch , avg_val_acc))
        best_val_acc = avg_val_acc
        best_epoch = epoch
        torch.save(model.state_dict() , 'best_model.pth')

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(avg_train_losses, label='Training Loss')
plt.plot(avg_val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(avg_train_accs, label='Training Accuracy')
plt.plot(avg_val_accs, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('training_validation_metrics.png')
plt.show()
print('Best model saved at epoch {} , acc : {:.4f}'.format(best_epoch , best_val_acc))