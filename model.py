import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)  # Dropout layer with a probability of 0.2

    def forward(self, x):
        output = self.l1(x)
        output = self.relu(output)
        output = self.dropout(output) #to prevent overfitting
        output = self.l2(output)
        #not using activation like softmax bc we will use nn.CrossEntropyLoss which does that for us
        return output