import json
from nltk_utils import stem, tokenize, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = [] 
tags = [] #json tags
xy = [] #will hold patterns and tags

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)#extend not append bc we dont want to put an array of arrays
        xy.append((w,tag)) #add a tuple of w,tag

ignore_words = ['?', '!', ',' , '.']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

#create bag of words
X_train = []
y_train = []
for(pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) #Cross entropy loss, so we dont care for 1 hot encoding

X_train = np.array(X_train)
y_train = np.array(y_train)

#new data set
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = torch.tensor(X_train, dtype=torch.float32)  # Convert to PyTorch tensor
        self.y_data = torch.tensor(y_train, dtype=torch.long)     # Convert to PyTorch tensor

    def __getitem__(self, index):  # Fixed method name
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    

#hyperparameters
batch_size = 8
hidden_size = 64
output_size = len(tags)
input_size = len(X_train[0])#length of each bag of words we made
learning_rate = 0.001
num_epochs = 1000
#print(input_size, len(all_words)) #making sure it all matches up
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
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete, file saved to {FILE}')