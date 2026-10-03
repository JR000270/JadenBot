import json
from embeddings import encode, ENCODER_NAME
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

tags = [] #json tags
xy = [] #will hold patterns and tags

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        xy.append((pattern,tag)) #raw sentence, the transformer does its own tokenizing

tags = sorted(set(tags))

#turn every pattern into a 384 number embedding in one batch (much faster than one at a time)
X_train = encode([pattern for (pattern, tag) in xy])
y_train = np.array([tags.index(tag) for (pattern, tag) in xy]) #Cross entropy loss, so we dont care for 1 hot encoding

#new data set
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        # Convert to PyTorch tensors
        self.x_data = torch.tensor(X_train, dtype=torch.float32)  
        self.y_data = torch.tensor(y_train, dtype=torch.long)     

    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    

#hyperparameters
batch_size = 8
hidden_size = 64
output_size = len(tags)
input_size = X_train.shape[1]#length of each embedding (384 for MiniLM)
learning_rate = 0.001
num_epochs = 1000
#print(input_size) #making sure it all matches up
#print(output_size, len(tags))

dataset = ChatDataset()
train_loader = DataLoader(dataset= dataset, batch_size= batch_size, shuffle= True, num_workers= 0) #num workers is multithreading/processing. Makes the loading a little faster

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #if we have a gpu, use it. else use cpu
model = NeuralNet(input_size, hidden_size, output_size).to(device) #put the model on the device

#loss and optimizer
criterion = nn.CrossEntropyLoss() #loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #adam optimizer

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)  # Ensure tensors are on the same device as the model
        labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
print(f'Final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "encoder_name": ENCODER_NAME, #so chat.py can check it's using the same encoder
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete, file saved to {FILE}')