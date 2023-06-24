import time
import matplotlib.pyplot as plt
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import os
# Disable parallelism and avoid the warning message
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define model IDs
model_ids = ['google/flan-t5-large', 'google/flan-t5-xl']

# Define prompts and types
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
    'The center of Tropical Storm Arlene, at 02/1800 UTC, is near 26.7N 86.2W. This position is about 425 km/230 nm to the west of Fort Myers in Florida, and it is about 550 km/297 nm to the NNW of the western tip of Cuba. The tropical storm is moving southward, or 175 degrees, 4 knots. The estimated minimum central pressure is 1002 mb. The maximum sustained wind speeds are 35 knots with gusts to 45 knots. The sea heights that are close to the tropical storm are ranging from 6 feet to a maximum of 10 feet.  Precipitation: scattered to numerous moderate is within 180 nm of the center in the NE quadrant. Isolated moderate is from 25N to 27N between 80W and 84W, including parts of south Florida.  Broad surface low pressure extends from the area of the tropical storm, through the Yucatan Channel, into the NW part of the Caribbean Sea.   Where and when will the storm make landfall?'
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

# Create empty lists to store generation times, model load times, tokenizer load times, and pipeline load times
xl_generation_times = []
large_generation_times = []

xl_model_load_times = []
large_model_load_times = []

xl_tokenizer_load_times = []
large_tokenizer_load_times = []

xl_pipeline_load_times = []
large_pipeline_load_times = []

prompt_types = []

for model_id in model_ids:
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
    if model_id == 'google/flan-t5-large':
        large_model_load_times.append(model_end_time - model_start_time)
        large_tokenizer_load_times.append(tokenizer_end_time - tokenizer_start_time)
        large_pipeline_load_times.append(pipe_end_time - pipe_start_time)
    elif model_id == 'google/flan-t5-xl':
        xl_model_load_times.append(model_end_time - model_start_time)
        xl_tokenizer_load_times.append(tokenizer_end_time - tokenizer_start_time)
        xl_pipeline_load_times.append(pipe_end_time - pipe_start_time)

    # Print model results
    print()
    print(f"Results for model {model_id}")
    print("=" * 30)

    # Knowledge retrieval examples
    for i, prompt in enumerate(prompts):
        start_time = time.time()
        answer = local_llm(prompt)
        end_time = time.time()
        print(f"Prompt: {prompt}")
        print(f"Answer: {answer}")
        print(f"Generation Time: {end_time - start_time:.5f} seconds")
        print(f"Type: {types[i]}")
        print()

        prompt_types.append(types[i])  # Store the prompt type

        if model_id == 'google/flan-t5-large':
            large_generation_times.append(end_time - start_time)
        elif model_id == 'google/flan-t5-xl':
            xl_generation_times.append(end_time - start_time)

# Plot generation times
plt.figure(figsize=(18, 6))
plt.subplot(131)
plt.barh(range(len(types)), xl_generation_times, height=0.4, align='center', color='blue', label='XL Model')
plt.barh([x + 0.4 for x in range(len(types))], large_generation_times, height=0.4, align='center', color='orange', alpha=0.5, label='Large Model')
plt.yticks(range(len(types)), types)
plt.ylabel('Type')
plt.xlabel('Generation Time (seconds)')
plt.title('Generation Time Comparison')
plt.legend()

# Plot model load times
model_load_times = [sum(xl_model_load_times), sum(large_model_load_times)]
model_labels = ['XL Model', 'Large Model']

plt.subplot(132)
plt.bar(model_labels, model_load_times, color=['blue', 'orange'])
plt.ylabel('Load Time (seconds)')
plt.xlabel('Model')
plt.title('Model Load Time Comparison')

# Plot tokenizer load times
tokenizer_load_times = [sum(xl_tokenizer_load_times), sum(large_tokenizer_load_times)]

plt.subplot(133)
plt.bar(model_labels, tokenizer_load_times, color=['blue', 'orange'])
plt.ylabel('Load Time (seconds)')
plt.xlabel('Model')
plt.title('Tokenizer Load Time Comparison')

# Plot pipeline load times
pipeline_load_times = [sum(xl_pipeline_load_times), sum(large_pipeline_load_times)]

plt.figure(figsize=(8, 6))
plt.bar(model_labels, pipeline_load_times, color=['blue', 'orange'])
plt.ylabel('Load Time (seconds)')
plt.xlabel('Model')
plt.title('Pipeline Load Time Comparison')

plt.tight_layout()
plt.show()

