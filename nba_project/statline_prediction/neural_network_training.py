import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.optim as optim
import os


data = pd.read_csv('player_data.csv')   

# Assuming you have a Pandas DataFrame 'data' with features and target variables
# Features are all columns except points, rebounds, assists
X = data.drop(['points', 'rebounds', 'assists'], axis=1).values
y = data[['points', 'rebounds', 'assists']].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the Neural Network model
class StatlineNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StatlineNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Model Parameters
input_size = X_train.shape[1]  # Number of input features
hidden_size = 64               # You can adjust this number
output_size = 3                # Predicting points, rebounds, assists

model = StatlineNN(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.L1Loss()  # Mean Absolute Error
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Training the model
num_epochs = 50

# Load the best loss from the model info text doc

if not os.path.exists('model_info.txt'):
    best_loss = float('inf')
else:
    with open('model_info.txt', 'r') as file:
        best_loss = float(file.read())




for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)
    if average_loss < best_loss:
        best_loss = average_loss

        with open('model_info.txt', 'w') as file:
            file.write(str(best_loss))
            
        torch.save(model.state_dict(), 'best_model_weights.pth')  # Save the weights

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

# Evaluate the model
model.eval()  # Set the model to evaluation mode
y_pred = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        outputs = model(X_batch)
        y_pred.append(outputs)

y_pred = torch.cat(y_pred).numpy()

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae:.4f}')
