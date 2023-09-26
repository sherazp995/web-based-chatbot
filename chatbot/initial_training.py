import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Replace 'your_file_path' with the actual file path
file_path = 'chatbot.txt'

def train():
    with open(file_path, "r") as file:
        training_data = file.read()

    # Tokenize the training data
    input_ids = tokenizer.encode(training_data, return_tensors="pt", truncation=True)
    attention_mask = torch.ones_like(input_ids)

    # Fine-tune the GPT-2 model on your data
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model.train()
    optimizer.zero_grad()

    loss = model(input_ids, attention_mask=attention_mask, labels=input_ids).loss
    loss.backward()
    optimizer.step()

try:
    file_size = os.path.getsize(file_path)
except FileNotFoundError:
    print("File not found.")
    file_size = 0
except OSError:
    print("OS error occurred.")
    file_size = 0

if(file_size > 0):
    train()
    print("Training model")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_chatbot")
tokenizer.save_pretrained("fine_tuned_chatbot")
