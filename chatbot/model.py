import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("fine_tuned_chatbot")
tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_chatbot")

# Replace 'your_file_path' with the actual file path
file_path = 'chatbot.txt'

def retrain():
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

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_chatbot")
    tokenizer.save_pretrained("fine_tuned_chatbot")

try:
    file_size = os.path.getsize(file_path)
except FileNotFoundError:
    print("File not found.")
    file_size = 0
except OSError:
    print("OS error occurred.")
    file_size = 0

if(file_size > 0):
    retrain()
    print("Retraining model")

# Function to generate a response based on user input
def generate_response(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors="pt", truncation=True)
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

print("Hi I am Bot, a pretrained model. Ask any question, for exiting write 'bye'")
# Chat with the chatbot
while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'bye':
        print("Chatbot: Goodbye!")
        break

    response = generate_response(user_input)
    print("Chatbot:", response)
