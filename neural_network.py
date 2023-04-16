# Imports
# all the modules necessary for model development and training
import torch
import torch.nn as nn

# for optimizers like SGD, adam, etc.
import torch.optim as optim

# this is for activation functions like ReLu, tanh etc.
import torch.nn.functional as F

# for better dataset management, like making mini batches
from torch.utils.data import DataLoader

# for loading standard datasets
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# create fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):  # pass the arguments to be used in model over here
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# sanity check of the model
model = NN(784, 10)
x = torch.randn(64, 784)
print(model.forward(x).shape)

# setting the GPU as computing device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Hyperparameter declaration
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

# loading dataset
train_dataset = datasets.MNIST(root = 'datasets/', train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root = 'datasets/', train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# initialize the network for computation
model = NN(input_size = input_size, num_classes = num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# training of network
for epoch in range(num_epochs):
    print(f'Running epoch number {epoch+1}')
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # reshaping before feeding to first hidden layer
        data = data.reshape(data.shape[0], -1)

        # forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or Adam step
        optimizer.step()

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('checking accuracy on train data')
    else:
        print('checking accuracy on test data')
    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)

            x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'The accuracy is : {(float(num_correct)/float(num_samples))*100:.2f}%')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)




