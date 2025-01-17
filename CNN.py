#Practice 1234567890
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
from pathlib import Path

#converting MNISt file into tensor of 4d(# of img,color,height,width)
#Note MNIST dataset is already accessable if u download Tensorflowlib on ur vscodepython env, etc.

transform = transforms.ToTensor()

#train_data
train_data = datasets.wake_vision

#test
test_data = datasets.MNIST(root='data_CNN', train=False, download=True, transform=transform)

#creating a batch of images.. depends on how many do you want.. for me,a'll go with 20
train_loader = DataLoader(train_data, batch_size=20, shuffle=True) #we're going to shuffle the train_data
test_loader = DataLoader(test_data, batch_size=20, shuffle=False)

#defining CNN and convolutional layer(mine, I'm making 2)
conv1 = nn.Conv2d(1, 6, 3,1,)#2dimensional, (1=input size, 6=filters, 3=kernerl size, 1=stride at a time)
conv2 = nn.Conv2d(6,16,3,1) #in conv2 u can decide whatever size and filter u want, same kernel and stride

#grab an image in MNIST 
for i, (X_train, y_train) in enumerate(train_data):
    break
x = X_train.view(1,1,28,28)
#performing Convolution
x = F.relu(conv1(x))
# print(x.shape) #torch.Size([1, 6, 26, 26]), 1=image, 6=filters and 26x26 matrix
#pooling
x = F.max_pool2d(x,2,2)
# print(x.shape) torch.Size([1, 6, 13, 13]), due to pooling the 26x26 matrix will be divide by 2x2

#2nd conv layer

x = F.relu(conv2(x))
# print(x.shape) the padding didn't set so we're lossing 2 pixles around the image torch.Size([1, 6, 13, 13])

#2nd Pooling layers
x = F.max_pool2d(x,2,2)
# print(x.shape), torch.Size([1, 16, 5, 5]), 11/2 is 5.5 but the mahine rounded down can't invent the data to round up

#Model Class
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        #the fully conected layer
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)
        
    #forward function
    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2) # 2x2 kernel with 2 as stride
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2) # 2x2 kernel with 2 as stride
        
        X = X.view(-1,16*5*5)  #putting negative to vary the batch size
        
        #Fully connected layer
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X,dim=1)
    
    
    # create instance of the model
    torch.manual_seed(41) 
model = ConvolutionalNetwork()
print(model)
        
#loss optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # the smaller lr the longer to train

start_time = time.time()
# create variables to track thoings out
epochs = 5
train_losses = [] 
test_losses = []
train_correct = []
test_correct = []


#Epochs loop
for i in range(epochs):
    trn_corr=0
    tst_corr=0
    
 
 
    #train
    for b, (X_train, y_train) in enumerate(train_loader):
        b+=1
        y_pred = model(X_train) # get predicted values from the training set
        loss = criterion(y_pred, y_train) #compare predictions to correct answer in y_train
        
        predicted = torch.max(y_pred.data, 1)[1] #add up to correct predictions. 
        batch_corr = (predicted == y_train).sum() #correct answer from this batch. True=1, False=0
        trn_corr += batch_corr #keep trach along with training
        
        
        
    #parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
     
    #results
        if b%600 == 0:
            print(f'Epoch: {i} Batch: {b} Loss: {loss.item()}')
            
    train_losses.append(loss)
    train_correct.append(trn_corr)
    
    
    #test

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate (test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1] # adding correct predictions
            tst_corr += (predicted == y_test).sum()
            
    loss = criterion(y_val,y_test)      
    test_losses.append(loss)
    test_correct.append(tst_corr)  




current_time = time.time()
total = current_time - start_time
print(f'Training Took: {total/60} minutes!')

# #graphing the loss
# train_losses = [tl.item() for tl in train_losses]
# plt.plot(train_losses, label = "traning Loss")
# plt.plot(test_losses, label = "validation Loss")
# plt.title("Loss at Epoch")
# plt.legend()


# Test the CNN after training

model.eval()  # Set the model to evaluation mode (important for dropout/batchnorm layers if used)
test_correct = 0
test_total = 0

with torch.no_grad():  # No gradients required for validation
    for X_test, y_test in test_loader:
        y_pred = model(X_test)  # Get predictions from the model
        _, predicted = torch.max(y_pred, 1)  # Get the class with the highest predicted probability
        test_total += y_test.size(0)  # Increment total number of samples
        test_correct += (predicted == y_test).sum().item()  # Count correct predictions

# Calculate accuracy
test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Visualizing test images and their predictions

# Switch to evaluation mode to disable any behavior like dropout (if used)
model.eval()

# Fetch some test samples and their labels
data_iter = iter(test_loader)
images, labels = next(data_iter)

# Get predictions
outputs = model(images)
_, predictions = torch.max(outputs, 1)

# Plot a few images with their true and predicted labels
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    ax = axes[i]
    ax.imshow(images[i].squeeze(), cmap='gray')
    ax.set_title(f"True: {labels[i].item()}, Pred: {predictions[i].item()}")
    ax.axis('off')

plt.show()

