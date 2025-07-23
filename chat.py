import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    try:
        intents = json.load(f)
    except json.JSONDecodeError:
        print("Error: Failed to load intents.json. Please check the file format.")
        exit()

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "JadenBot"

def get_response(tmsg):
    sentence = tokenize(tmsg)  # Use the input message
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device, dtype=torch.float32)

    output = model(X)
    _, pred = torch.max(output, dim=1)
    tag = tags[pred.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "I don't have a response for this yet, I be primitive! Maybe I can help you with something else?"

