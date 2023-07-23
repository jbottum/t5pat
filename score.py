# Import necessary libraries
import time
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, HuggingFacePipeline
import transformers
import bert_score
import logging
import os
import json
import csv
import pandas as pd
from pprint import pprint
import datetime

# Define sample model data using Python variables and lists
user_id = 'jbottum'
project_id = 'project1'
model_id = 'google/flan-t5-large'

# Define prompts and question types
prompts = [
    # List of prompts
]

types = [
    # List of question types corresponding to each prompt
]

# Example reference answers
reference_list = [
    # List of reference answers corresponding to each prompt
]

# Load sample model data into a Python dictionary
data = {
    "user_id": user_id,
    "project_id": project_id,
    "model_id": model_id,
    "prompts": prompts,
    "types": types,
    "reference_list": reference_list
}

# Save data as JSON
with open("data.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

# Transpose the data to convert it to CSV format
transposed_data = list(zip(prompts, types, reference_list))

# Save data as CSV
with open("data.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Prompts", "Types", "Reference List"])
    writer.writerows(transposed_data)

# Load data from JSON file into dictionary
with open("data.json", "r") as json_file:
    data = json.load(json_file)

# Load tokenizer
tokenizer_start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer_end_time = time.time()

# Load model
model_start_time = time.time()
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
model_end_time = time.time()

# Load pipeline
pipe_start_time = time.time()
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
local_llm = HuggingFacePipeline(pipeline=pipe)
pipe_end_time = time.time()

# Store loading times
model_load_time = model_end_time - model_start_time
tokenizer_load_time = tokenizer_end_time - tokenizer_start_time
pipeline_load_time = pipe_end_time - pipe_start_time

# Create the pipeline for text generation
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=50)

# Initialize the dictionary to store results
epoch_record = {}

# Loop through the prompts and generate text
for i, prompt in enumerate(data['prompts']):
    start_time = time.time()
    generated = pipe(prompt, num_return_sequences=1)[0]['generated_text']
    end_time = time.time()
    generation_time = end_time - start_time

    # Store the results in a dictionary record
    record = {
        'timestamp': start_time,
        'project_name': data['project_id'],
        'user_id': data['user_id'],
        'model_id': data['model_id'],
        'prompt': prompt,
        'type': data['types'][i],
        'reference_answer': data['reference_list'][i],
        'generated_answer': generated,
        'generation_time': generation_time,
        'model_load_time': model_load_time,
        'tokenizer_load_time': tokenizer_load_time,
        'pipeline_load_time': pipeline_load_time,
    }

    # Calculate BERTScore
    P, R, F1 = bert_score.score([generated], [record['reference_answer']], lang="en", verbose=False)
    record['precision'] = P.numpy()[0].item()
    record['recall'] = R.numpy()[0].item()
    record['f1'] = F1.numpy()[0].item()

    # Store the record in epoch_record with a unique key
    epoch_record[f"Example {i + 1}"] = record

# Print the results in the same format as the current script
print()
print(f"Model: {data['model_id']}")
print(f"Project: {data['project_id']}")
print(f"User: {data['user_id']}")
print("Run Date:", datetime.datetime.fromtimestamp(record['timestamp']).strftime('%Y-%m-%d %H:%M:%S'))
print("Model load time:", round(record['model_load_time'], 5))
print("Tokenizer load time:", round(record['tokenizer_load_time'], 5))
print("Pipeline load time:", round(record['pipeline_load_time'], 5))
print("=" * 30)

for example_number, record in epoch_record.items():
    print(f"{example_number}:")
    print("Prompt:", record['prompt'])
    print("Generated Text:", record['generated_answer'])
    print("Reference Answer:", record['reference_answer'])
    print("Generation Time:", round(record['generation_time'], 5))
    print("Type:", record['type'])
    print("Precision:", round(record['precision'], 5))
    print("Recall:", round(record['recall'], 5))
    print("F1 Score:", round(record['f1'], 5))
    print("=" * 10)

# Save epoch_record as a CSV file
csv_file = "epoch_record.csv"
with open(csv_file, "w", newline="") as csvfile:
    fieldnames = [
        'timestamp', 'project_name', 'user_id', 'model_id', 'prompt', '

