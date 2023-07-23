# !pip install einops
# !pip install transformers
# !pip install --upgrade transformers
# !pip install accelerate
# !pip install torch
# !pip install langchain
# !pip install bert-score

import time
import matplotlib.pyplot as plt
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import transformers
import torch

import bert_score
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


import os
# Disable parallelism and avoid the warning message
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# create a sample model data using python variables and lists
user_id = 'jbottum'
project_id = 'project1'
model_id = 'google/flan-t5-large'

# Define prompts and question types
prompts = [
    'What is the capital of Germany?',
    'What is the capital of Spain?',
    'What is the capital of Canada?',
    'What is the next number in the sequence: 2, 4, 6, 8, ...? If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?',
    'If you eat too much junk food, what will happen to your health? How does smoking affect the risk of lung cancer?',
    'In the same way that pen is related to paper, what is fork related to? If tree is related to forest, what is brick related to?',
    'Every time John eats peanuts, he gets a rash. Does John have a peanut allergy? Every time Sarah studies for a test, she gets an A. Will Sarah get an A on the next test if she studies?',
    'All dogs have fur. Max is a dog. Does Max have fur? If it is raining outside, and Mary does not like to get wet, will Mary take an umbrella?',
    'If I had studied harder, would I have passed the exam? What would have happened if Thomas Edison had not invented the light bulb?',
    'The center of Tropical Storm Arlene, at 02/1800 UTC, is near 26.7N 86.2W. This position is about 425 km/230 nm to the west of Fort Myers in Florida, and it is about 550 km/297 nm to the NNW of the western tip of Cuba. The tropical storm is moving southward, or 175 degrees, 4 knots. The estimated minimum central pressure is 1002 mb. The maximum sustained wind speeds are 35 knots with gusts to 45 knots. The sea heights that are close to the tropical storm are ranging from 6 feet to a maximum of 10 feet.  Precipitation: scattered to numerous moderate is within 180 nm of the center in the NE quadrant. Isolated moderate is from 25N to 27N between 80W and 84W, including parts of south Florida.  Broad surface low pressure extends from the area of the tropical storm, through the Yucatan Channel, into the NW part of the Caribbean Sea.   Where and when will the storm make landfall?',
]

types = [
    'Knowledge Retrieval',
    'Knowledge Retrieval',
    'Knowledge Retrieval',
    'Logical Reasoning',
    'Cause and Effect',
    'Analogical Reasoning',
    'Inductive Reasoning',
    'Deductive Reasoning',
    'Counterfactual Reasoning',
    'In Context'
]

# Example reference answers
reference_list = [
    'The capital of Germany is Berlin',
    'The capital of Spain is Madrid',
    'The capital of Canada is Ottawa',
    'The next number in the sequence is 10.  Yes, Fluffy is a cat and therefore has a tail.',
    'Eating junk food can result in health problems like weight gain and high cholesterol. Smoking can cause lung issues including cancer.',
    'Fork is related to a plate.  A brick is related to a building.',
    'Maybe, to determine if Johns rash is caused by peanuts, he should take an allergy test for peanuts.   Maybe, Sarah will likely do well if she studies and she may be able to get an A.',
    'Yes, Max is a dog and has fur.   Yes, Mary will take an umbrella.',
    'Yes, if you studied harder, you would have passed the test.  If Thomas Edison did not invent the light blub, another inventor would have created the light bulb.',
    'If Arlene continues in the same direciton and speed, storm will make landfall in the Forida Keys in 18 hours from this report.'
]



# load sample model data into a python dictionary
data = {
    "user_id": user_id,
    "project_id": project_id,
    "model_id": model_id,
    "prompts": prompts,
    "types": types,
    "reference_list": reference_list
}

# load the sample data into a JSON File

import json

# Save data as JSON
with open("data.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

# load the sample data into a CSV file

import csv

# Transpose the data to convert it to CSV format
transposed_data = list(zip(prompts, types, reference_list))

# Save data as CSV
with open("data.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Prompts", "Types", "Reference List"])
    writer.writerows(transposed_data)

# Load data from JSON file into Dictionary

import json

# Load data from the JSON file into a dictionary
with open("data.json", "r") as json_file:
    data = json.load(json_file)

# Print the dictionary to check the loaded data
print(data)

import time
import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline, HuggingFacePipeline

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

import time
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, pipeline
import bert_score

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
import time
import datetime
from transformers import pipeline
import bert_score
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

import csv
import json

# Save epoch_record as a CSV file
csv_file = "epoch_record.csv"
with open(csv_file, "w", newline="") as csvfile:
    fieldnames = [
        'timestamp', 'project_name', 'user_id', 'model_id', 'prompt', 'type',
        'reference_answer', 'generated_answer', 'generation_time',
        'model_load_time','tokenizer_load_time','pipeline_load_time',
        'precision', 'recall', 'f1'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for example_number, record in epoch_record.items():
        writer.writerow(record)

# Save epoch_record as a JSON file
json_file = "epoch_record.json"
with open(json_file, "w") as jsonfile:
    json.dump(epoch_record, jsonfile, indent=4)


import pandas as pd

# Read the CSV file
csv_file = "epoch_record.csv"
df = pd.read_csv(csv_file)

# Display the DataFrame with headers
df

import json
from pprint import pprint

# Read the JSON file
json_file = "epoch_record.json"
with open(json_file, "r") as jsonfile:
    json_data = json.load(jsonfile)

# Display the JSON data with formatting
pprint(json_data)

