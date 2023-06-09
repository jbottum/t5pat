import time
import matplotlib.pyplot as plt
from langchain.llms import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import transformers
import torch

import os
# Disable parallelism and avoid the warning message
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define model IDs
# model_ids = ['google/flan-t5-large', 'google/flan-t5-xl', 'tiiuae/falcon-7b']
model_ids = ['tiiuae/falcon-7b']
# model_ids = ['google/flan-t5-xl']

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

# Create empty lists to store generation times, model load times, tokenizer load times, and pipeline load times
xl_generation_times = []
large_generation_times = []
falcon_generation_times = []

xl_model_load_times = []
large_model_load_times = []
falcon_model_load_times = []

xl_tokenizer_load_times = []
large_tokenizer_load_times = []
falcon_tokenizer_load_times = []

xl_pipeline_load_times = []
large_pipeline_load_times = []
falcon_pipeline_load_times = []

prompt_types = []

for model_id in model_ids:
    # Load tokenizer
    tokenizer_start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer_end_time = time.time()

    # Load model
    model_start_time = time.time()
    if model_id == 'tiiuae/falcon-7b':
      model = AutoModelForCausalLM.from_pretrained(model_id)
    else:
      model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model_end_time = time.time()

    # Load pipeline
    pipe_start_time = time.time()
    if model_id == 'tiiuae/falcon-7b':
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
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
    elif model_id == 'tiiuae/falcon-7b':
        falcon_model_load_times.append(model_end_time - model_start_time)
        falcon_tokenizer_load_times.append(tokenizer_end_time - tokenizer_start_time)
        falcon_pipeline_load_times.append(pipe_end_time - pipe_start_time)

    # Print model results
    print()
    print(f"Results for model: {model_id}")
    print("=" * 30)

    # Knowledge retrieval examples
    if model_id == 'tiiuae/falcon-7b':
        sequences = pipeline(
            "Draft an apology email to a customer who experienced poor service.",
            num_return_sequences=3,
            max_length=100,
            min_length=10,
        )
        for i, seq in enumerate(sequences):
            print(f"Retrieval Example {i + 1}:")
            print(seq["generated_text"])
            print("=" * 10)

    # Generation examples
    for i, prompt in enumerate(prompts):
        start_time = time.time()
        if model_id == 'tiiuae/falcon-7b':
            generated = pipeline(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        else:
            generated = pipe(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        end_time = time.time()
        generation_time = end_time - start_time

        # Store generation times
        if model_id == 'google/flan-t5-large':
            large_generation_times.append(generation_time)
        elif model_id == 'google/flan-t5-xl':
            xl_generation_times.append(generation_time)
        elif model_id == 'tiiuae/falcon-7b':
            falcon_generation_times.append(generation_time)

        # Print generation example
        print(f"Generation Example {i + 1}:")
        print("Prompt:", prompt)
        print("Generated Text:", generated)
        print("=" * 10)

        prompt_types.append(types[i])

# Print average loading and generation times
print()
print("Average Loading Times:")
print("Flan T5-Large Tokenizer:", sum(large_tokenizer_load_times) / len(large_tokenizer_load_times), "seconds")
print("Flan T5-Large Model:", sum(large_model_load_times) / len(large_model_load_times), "seconds")
print("Flan T5-Large Pipeline:", sum(large_pipeline_load_times) / len(large_pipeline_load_times), "seconds")
print()
print("Flan T5-XL Tokenizer:", sum(xl_tokenizer_load_times) / len(xl_tokenizer_load_times), "seconds")
print("Flan T5-XL Model:", sum(xl_model_load_times) / len(xl_model_load_times), "seconds")
print("Flan T5-XL Pipeline:", sum(xl_pipeline_load_times) / len(xl_pipeline_load_times), "seconds")
print()
print("Falcon-7B Tokenizer:", sum(falcon_tokenizer_load_times) / len(falcon_tokenizer_load_times), "seconds")
print("Falcon-7B Model:", sum(falcon_model_load_times) / len(falcon_model_load_times), "seconds")
print("Falcon-7B Pipeline:", sum(falcon_pipeline_load_times) / len(falcon_pipeline_load_times), "seconds")
print()
print("Average Generation Times:")
print("Flan T5-Large:", sum(large_generation_times) / len(large_generation_times), "seconds")
print("Flan T5-XL:", sum(xl_generation_times) / len(xl_generation_times), "seconds")
print("Falcon-7B:", sum(falcon_generation_times) / len(falcon_generation_times), "seconds")

# Generate plot for generation times
plt.figure(figsize=(12, 6))
plt.plot(prompt_types, xl_generation_times, label='Flan T5-XL')
plt.plot(prompt_types, large_generation_times, label='Flan T5-Large')
plt.plot(prompt_types, falcon_generation_times, label='Falcon-7B')
plt.xlabel('Prompt Type')
plt.ylabel('Generation Time (seconds)')
plt.title('Generation Time by Prompt Type')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Generate plot for loading times
plt.figure(figsize=(12, 6))
plt.bar(['Flan T5-Large', 'Flan T5-XL', 'Falcon-7B'], [sum(large_model_load_times) / len(large_model_load_times),
                                                        sum(xl_model_load_times) / len(xl_model_load_times),
                                                        sum(falcon_model_load_times) / len(falcon_model_load_times)],
        label='Model Load Time')
plt.bar(['Flan T5-Large', 'Flan T5-XL', 'Falcon-7B'], [sum(large_tokenizer_load_times) / len(large_tokenizer_load_times),
                                                        sum(xl_tokenizer_load_times) / len(xl_tokenizer_load_times),
                                                        sum(falcon_tokenizer_load_times) / len(falcon_tokenizer_load_times)],
        label='Tokenizer Load Time')
plt.bar(['Flan T5-Large', 'Flan T5-XL', 'Falcon-7B'], [sum(large_pipeline_load_times) / len(large_pipeline_load_times),
                                                        sum(xl_pipeline_load_times) / len(xl_pipeline_load_times),
                                                        sum(falcon_pipeline_load_times) / len(falcon_pipeline_load_times)],
        label='Pipeline Load Time')
plt.xlabel('Model')
plt.ylabel('Loading Time (seconds)')
plt.title('Average Loading Time by Model')
plt.legend()
plt.tight_layout()
plt.show()

