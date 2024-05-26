import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from Data.Data_Factory_v2 import *
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from sklearn.metrics import classification_report, accuracy_score

'''
Train a MNIST classifier for evaluation
'''
#parameter setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
epoch_num = 50
#use resnet18 as pretrained model
resnet18 = models.resnet18(pretrained = True).to(device)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 10).to(device)
resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)
#print(resnet18.conv1)

#load data
all_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
data_train = FastMNIST(root='../Data/Data_Store', train=True, transform=all_transforms)
dataloader_train = DataLoader(data_train, batch_size = 200, shuffle=True)
data_test = FastMNIST(root='../Data/Data_Store', train=False, transform=all_transforms)
dataloader_test = DataLoader(data_test, batch_size = 200, shuffle=True)

loss = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(resnet18.parameters(), lr=0.001)

# Training loop
for epoch in range(epoch_num):
    resnet18.train()
    start_time = time.time()
    running_loss = 0.0
    for i, data in enumerate(dataloader_train, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet18(inputs)
        loss_value = loss(outputs, labels)
        loss_value.backward()
        optimizer.step()

        running_loss += loss_value.item()
    correct = 0
    total = 0
    resnet18.eval()
    with torch.no_grad():
        for data in dataloader_test:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = resnet18(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end_time = time.time()
    passing_minutes = int((end_time - start_time) // 60)
    passing_seconds = int((end_time - start_time) % 60)
    print(f'Epoch {epoch+1}, Loss: {running_loss}, Accuracy: {100 * correct / total}%, time passing:{passing_minutes}m{passing_seconds}s.')
print('Finished Training')

#Final evaluation
test_images = data_test.data
test_images = test_images.to(device)
test_labels = data_test.target
test_labels = test_labels.to(device)
resnet18.eval()
with torch.no_grad():
    outputs = resnet18(test_images)
    _, predicted = torch.max(outputs.data, 1)
    cla_report = classification_report(test_labels.cpu().numpy(), predicted.cpu().numpy(), digits=3)
    acc = accuracy_score(test_labels.cpu().numpy(), predicted.cpu().numpy())
    print(cla_report)

cur_time = int(time.time())
torch.save(resnet18, f'../SavedModels/classifiers/resnet18_mnist_f{cur_time}_acc{int(acc * 1000)}.pth')
