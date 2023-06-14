from llama_index import LLMPredictor, ServiceContext, SimpleDirectoryReader, GPTListIndex
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.embeddings import SentenceTransformerEmbeddings

# Load models and pipelines
model_id = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
local_llm_g_flan_t5_large = HuggingFacePipeline(pipeline=pipe)

# Test Knowledge retrieval
questions = [
    'What is the capital of Germany?',
    'What is the capital of Spain?',
    'What is the capital of Canada?'
]

print("\nKnowledge Retrieval Examples")
print("=" * 30)

for question in questions:
    answer = local_llm_g_flan_t5_large(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")

# Indexing
directory_path = '/content/dir'  # Replace with the actual directory path
documents = SimpleDirectoryReader(directory_path).load_data()
embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
llm_predictor = LLMPredictor(llm=local_llm_g_flan_t5_large)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
index = GPTListIndex.from_documents(documents, service_context=service_context)

# Test Question and Answer
print("\nQuestion Answer Example, Text-to-Text")
print("=" * 30)

question1 = 'The center of Tropical Storm Arlene, at 02/1800 UTC, is near 26.7N 86.2W. This position is about 425 km/230 nm to the west of Fort Myers in Florida, and it is about 550 km/297 nm to the NNW of the western tip of Cuba. The tropical storm is moving southward, or 175 degrees, 4 knots. The estimated minimum central pressure is 1002 mb. The maximum sustained wind speeds are 35 knots with gusts to 45 knots. The sea heights that are close to the tropical storm are ranging from 6 feet to a maximum of 10 feet.  Precipitation: scattered to numerous moderate is within 180 nm of the center in the NE quadrant.  Isolated moderate is from 25N to 27N between 80W and 84W, including parts of south Florida.  Broad surface low pressure extends from the area of the tropical storm, through the Yucatan Channel, into the NW part of the Caribbean Sea.   Where and when will the storm make landfall?'

answer = local_llm_g_flan_t5_large(question1)
print(f"Question: {question1}")
print(f"Answer: {answer}\n")

print("\nQuestion Answer with Reasoning Examples")
print("=" * 30)

question2 = 'Logical Reasoning: What is the next number in the sequence: 2, 4, 6, 8, ...? If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?'

answer = local_llm_g_flan_t5_large(question2)
print(f"Question: {question2}")
print(f"Answer: {answer}\n")

question3 = 'Cause and Effect: If you eat too much junk food, what will happen to your health? How does smoking affect the risk of lung cancer?'

answer = local_llm_g_flan_t5_large(question3)
print(f"Question: {question3}")
print(f"Answer: {answer}\n")

question4 = 'Analogical Reasoning: In the same way that "pen" is related to "paper", what is "fork" related to? If "tree" is related to "forest", what is "brick" related to?'

answer = local_llm_g_flan_t5_large(question4)
print(f"Question: {question4}")
print(f"Answer: {answer}\n")

question5 = 'Deductive Reasoning: All dogs have fur. Max is a dog. Does Max have fur? If it is raining outside, and Mary doesn\'t like to get wet, will Mary take an umbrella?'

answer = local_llm_g_flan_t5_large(question5)
print(f"Question: {question5}")
print(f"Answer: {answer}\n")

question6 = 'Inductive Reasoning: Every time John eats peanuts, he gets a rash. Does John have a peanut allergy? Every time Sarah studies for a test, she gets an A. Will Sarah get an A on the next test if she studies?'

answer = local_llm_g_flan_t5_large(question6)
print(f"Question: {question6}")
print(f"Answer: {answer}\n")

question7 = "Counterfactual Reasoning: If I had studied harder, would I have passed the exam? What would have happened if Thomas Edison hadn't invented the light bulb"

answer = local_llm_g_flan_t5_large(question7)
print(f"Question: {question7}")
print(f"Answer: {answer}\n")

