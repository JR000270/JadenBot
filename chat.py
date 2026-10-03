import random
import json
import torch
from model import NeuralNet
from embeddings import encode, ENCODER_NAME

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    try:
        intents = json.load(f)
    except json.JSONDecodeError:
        print("Error: Failed to load intents.json. Please check the file format.")
        exit()

FILE = "data.pth"
data = torch.load(FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
tags = data["tags"]
model_state = data["model_state"]

#fail loudly instead of giving confident nonsense if data.pth was trained with a different encoder
if data.get("encoder_name") != ENCODER_NAME:
    raise RuntimeError(f"data.pth was trained with encoder {data.get('encoder_name')!r} but chat uses {ENCODER_NAME!r}. Re-run train.py.")

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "JadenBot"

def get_response(tmsg):
    X = encode([tmsg])  # list of one sentence -> shape (1, 384), the batch dimension the model expects
    X = torch.from_numpy(X).to(device, dtype=torch.float32)

    with torch.no_grad(): #only predicting, so skip tracking gradients
        output = model(X)
    _, pred = torch.max(output, dim=1)
    tag = tags[pred.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "I don't have a response for this yet, I'm primitive! Maybe I can help you with something else or rephrase that?"

