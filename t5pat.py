from llama_index import LLMPredictor, PromptHelper, ServiceContext, SimpleDirectoryReader, GPTVectorStoreIndex, GPTListIndex
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain.llms import HuggingFacePipeline 

model_id = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=512
)

local_llm_g_flan_t5_large = HuggingFacePipeline(pipeline=pipe)
print(local_llm_g_flan_t5_large('What is the capital of Germany? '))
print(local_llm_g_flan_t5_large('What is the capital of Spain? '))
print(local_llm_g_flan_t5_large('What is the capital of Canada? '))
# all is working untill here

from llama_index import GPTListIndex, SimpleDirectoryReader, GPTVectorStoreIndex
from langchain.embeddings import SentenceTransformerEmbeddings 
from llama_index import LangchainEmbedding, ServiceContext

directory_path_ = '/content/dir'
documents = SimpleDirectoryReader(directory_path_).load_data()

llm_predictor = LLMPredictor(llm=local_llm_g_flan_t5_large)
embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# service_context = ServiceContext.from_defaults(embed_model=embed_model, llm_predictor=llm_predictor)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
index = GPTListIndex.from_documents(documents, service_context=service_context)
print("Indexing completed successfully")
