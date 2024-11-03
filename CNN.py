import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.model_selection import KFold
from Read_NiftiData import train_dataset, test_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = None
        self.fc2 = nn.Linear(500, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x.float())))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 500).to(device)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, optimizer, criterion, scheduler, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        scheduler.step()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {100 * correct / total}%')

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def cross_validate(dataset, k_folds=5):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=val_subsampler)
        
        model = Net().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        
        train_model(model, train_loader, optimizer, criterion, scheduler)
        
        accuracy = evaluate_model(model, val_loader)
        print(f'Validation Accuracy for fold {fold}: {accuracy}%')
        print('--------------------------------')

# Combine train and test datasets for cross-validation
full_dataset = train_dataset + test_dataset

# Perform cross-validation
cross_validate(full_dataset)

# Final evaluation on test set
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)
final_model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
train_model(final_model, train_loader, optimizer, criterion, scheduler)

final_accuracy = evaluate_model(final_model, test_loader)
print(f'Final Test Accuracy: {final_accuracy}%')