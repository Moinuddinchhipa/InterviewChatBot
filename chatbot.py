import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the dataset
data = pd.read_csv('data_science_questions.csv')

# Data cleaning: Removing duplicates and NaNs
data.drop_duplicates(subset='question', inplace=True)
data.dropna(inplace=True)

# Preview the data
print(data.head())

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add padding if necessary (GPT-2 doesn't support padding by default)
tokenizer.pad_token = tokenizer.eos_token

   # Function to generate responses
def generate_response(question):
    # Encode the question and generate attention mask
    inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
    attention_mask = inputs['attention_mask']
    
    # Generate the response
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=attention_mask,
        max_length=150, 
        num_return_sequences=1, 
        do_sample=True
    )

    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Simple chatbot interaction
def chatbot():
    print("Chatbot: Hi! I'm here to help you prepare for your interview.")
    while True:
        question = input("You: ")
        if question.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = generate_response(question)
        print(f"Chatbot: {response}")

chatbot()