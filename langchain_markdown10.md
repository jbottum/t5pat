
# Testing LLMs with LangChain in a local environment for (7) types of reasoning

Within (30) minutes of reading this post, you should be able to complete model serving requests from two popular python-based large language models (LLM) using LangChain on your local computer without requiring the connection or costs to an external 3rd party API server, such as HuggingFaceHub or OpenAI.  We provide scripts that enable you to test these LLMs' capabilities to answer three types of prompts - Knowledge Retreival, Text2Text Question Answer and seven forms for Reasoning for Question Answer.  We will walk your through installing dependences, and we will review the code and the output.  As an optional exercise, we will show to build and run a docker container with this code.  Note - if you only have a few minutes and just want to run the models (and you have python3 and pip installed), you can proceed to [Step 1](#step-1---installing-dependencies-for-the-models-step1).

## Why run local

Some of the reasons why you may need to run your model locally, and not use an external API server, include:

* Security
    * You might want to fine tune the model and not post the derivative model on an external API server.
* Cost
    * You might want to avoid paying an external company for API calls. 
* Performance
    * You might need to manage your model's response times by using a private network and/or a specific server / processor type.
* Functionality
    * Your model might only run locally (i.e. Blenderbot, Meta's chatbot models).

This project provides the code and process to run two types of pretrained, large language models (Flan-T5-Large and Sentence-BERT, all-MiniLM-L6-v2) using LangChain on your local computer. We selected these top performing models because several developers were having trouble running a tutorial locally, as tracked in this github issue, [https://github.com/hwchase17/LangChain/issues/4438](https://github.com/hwchase17/langchain/issues/4438).  

## LLM #1 - Flan-T5-Large

First, we will show the process to run the Flan-T5-Large model.   This transformer model, open sourced from Google, is designed for natural language processing tasks and provides both text-to-text and text generation capabilities. It is based on the T5 (Text-To-Text Transfer Transformer) architecture and has 780M parameters.  This [paper](https://arxiv.org/pdf/2210.11416.pdf), which provides the following chart, claims that the Flan-T5-Large achieved a MMLU score of 45.1%, which is pretty good when compared to ChatGPT3's score of 43.9% (see page 10). It is a fairly popular model, which had 446,125 downloads last month. For more detailed information on this model’s background, performance and capabilities, please see this link on HuggingFaceHub, [https://huggingface.co/google/flan-t5-large](https://huggingface.co/google/flan-t5-large).  

![alt_text](image1.png "image_tooltip")

## LLM #2 - S BERT, all-MiniLM-L6-v2

Second, we will show a sentence-transformer framework, specifically the BertModel with model_name='all-MiniLM-L6-v2', which was also open sourced from Google.  This model maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.  This model has been extensively evaluated for embedded sentences (Performance Sentence Embeddings) and for embedded search queries & paragraphs (Performance Semantic Search). 

The model is a general purpose model and was trained with more than 1 billion training pairs. The **all-MiniLM-L6-v2** is relatively small (80MB) and fast, yet it still offers good quality.  It is a very popular model and had 2,674,926 downloads last month.  The text below is from this page, [https://www.sbert.net/docs/pretrained_models.html](https://www.sbert.net/docs/pretrained_models.html). For more detailed information on this model’s background, performance and capabilities, please see this link [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).  

## SBERT.net, Sentence-Transformers - Pretrained Models

We provide various pre-trained models. Using these models is easy:

**from** **sentence_transformers** **import** SentenceTransformer

model = SentenceTransformer('model_name')

All models are hosted on the [HuggingFace Model Hub](https://huggingface.co/sentence-transformers).

## Model Overview

The following table provides an overview of (selected) models. They have been extensively evaluated for their quality to embedded sentences (Performance Sentence Embeddings) and to embedded search queries & paragraphs (Performance Semantic Search).

The **all-*** models where trained on all available training data (more than 1 billion training pairs) and are designed as **general purpose** models. The **all-mpnet-base-v2** model provides the best quality, while **all-MiniLM-L6-v2** is 5 times faster and still offers good quality. Toggle _All models_ to see all evaluated models or visit [HuggingFace Model Hub](https://huggingface.co/models?library=sentence-transformers) to view all existing sentence-transformers models.

![alt_text](image2.png "image_tooltip")

## LangChain - What is it? Why use it?

The text in this section is from [https://python.LangChain.com/en/latest/index.html](https://python.langchain.com/en/latest/index.html) 

LangChain is a framework for developing applications powered by language models. We believe that the most powerful and differentiated applications will not only call out to a language model, but will also be:

1. _Data-aware_: connect a language model to other sources of data
2. _Agentic_: allow a language model to interact with its environment

The LangChain framework is designed around these principles.  This is the Python specific portion of the documentation. For a purely conceptual guide to LangChain, see [here](https://docs.langchain.com/docs/). For the JavaScript documentation, see [here](https://js.langchain.com/docs/). For concepts and terminology, please see [here](https://python.langchain.com/en/latest/getting_started/concepts.html).

## Modules

These modules are the core abstractions which we view as the building blocks of any LLM-powered application. For each module LangChain provides standard, extendable interfaces. LangChain also provides external integrations and even end-to-end implementations for off-the-shelf use. The docs for each module contain quickstart examples, how-to guides, reference docs, and conceptual guides.

The modules are (from least to most complex):

* [Models](https://python.langchain.com/en/latest/modules/models.html): Supported model types and integrations.
* [Prompts](https://python.langchain.com/en/latest/modules/prompts.html): Prompt management, optimization, and serialization.
* [Memory](https://python.langchain.com/en/latest/modules/memory.html): Memory refers to the state that is persisted between calls of a chain/agent.
* [Indexes](https://python.langchain.com/en/latest/modules/indexes.html): Language models become much more powerful when combined with application-specific data - this module contains interfaces and integrations for loading, querying and updating external data.
* [Chains](https://python.langchain.com/en/latest/modules/chains.html): Chains are structured sequences of calls (to an LLM or to a different utility).
* [Agents](https://python.langchain.com/en/latest/modules/agents.html): An agent is a Chain in which an LLM, given a high-level directive and a set of tools, repeatedly decides an action, executes the action and observes the outcome until the high-level directive is complete.
* [Callbacks](https://python.langchain.com/en/latest/modules/callbacks/getting_started.html): Callbacks let you log and stream the intermediate steps of any chain, making it easy to observe, debug, and evaluate the internals of an application.

## Use Cases

Best practices and built-in implementations for common LangChain use cases:

* [Autonomous Agents](https://python.langchain.com/en/latest/use_cases/autonomous_agents.html): Autonomous agents are long-running agents that take many steps in an attempt to accomplish an objective. Examples include AutoGPT and BabyAGI.
* [Agent Simulations](https://python.langchain.com/en/latest/use_cases/agent_simulations.html): Putting agents in a sandbox and observing how they interact with each other and react to events can be an effective way to evaluate their long-range reasoning and planning abilities.
* [Personal Assistants](https://python.langchain.com/en/latest/use_cases/personal_assistants.html): One of the primary LangChain use cases. Personal assistants need to take actions, remember interactions, and have knowledge about your data.
* [Question Answering](https://python.langchain.com/en/latest/use_cases/question_answering.html): Another common LangChain use case. Answering questions over specific documents, only utilizing the information in those documents to construct an answer.
* [Chatbots](https://python.langchain.com/en/latest/use_cases/chatbots.html): Language models love to chat, making this a very natural use of them.
* [Querying Tabular Data](https://python.langchain.com/en/latest/use_cases/tabular.html): Recommended reading if you want to use language models to query structured data (CSVs, SQL, dataframes, etc).
* [Code Understanding](https://python.langchain.com/en/latest/use_cases/code.html): Recommended reading if you want to use language models to analyze code.
* [Interacting with APIs](https://python.langchain.com/en/latest/use_cases/apis.html): Enabling language models to interact with APIs is extremely powerful. It gives them access to up-to-date information and allows them to take actions.
* [Extraction](https://python.langchain.com/en/latest/use_cases/extraction.html): Extract structured information from text.
* [Summarization](https://python.langchain.com/en/latest/use_cases/summarization.html): Compressing longer documents. A type of Data-Augmented Generation.
* [Evaluation](https://python.langchain.com/en/latest/use_cases/evaluation.html): Generative models are hard to evaluate with traditional metrics. One promising approach is to use language models themselves to do the evaluation.

As you can see from the previous section, LangChain includes many advanced features and it enables complex model processing.   In our example, we will use models, prompts, and pipelines for question answering, text-to-text, sequence-to-sequence, and text-generation.

## Getting started

In our example and process, we wanted to simplify the getting started.   We selected specific LLMs to run in the LangChain framework, which will run in a local environment i.e. in an older, Mac laptop with 16GB RAM without GPUs.   We anticipate that many developers can use this initially and then modify our choices for your requirements.   

## Step 0

This post assumes that users have a terminal emulator, python and Docker installed.  For those that do not, the installation for that software can be found in the instructions below.   Before installing the software, you should consider which directories that you will use.  Most dependencies will install automatically.   You will need a directory for the python script that runs the models and we suggest a directory named t5pat.  If you know how to access your terminal and have a recent version of Python 3.x and of Docker, then please skip to Step 1.

## Accessing your terminal

To access the terminal on a MacBook Air or any macOS device, you can follow these steps:

Click on the "Finder" icon in the Dock, which is typically located at the bottom of the screen.

In the Finder window, navigate to the "Applications" folder.

Inside the "Applications" folder, open the "Utilities" folder.

Look for an application called "Terminal" and double-click on it to launch the Terminal.

Alternatively, you can use Spotlight Search to quickly open the Terminal:

Press the "Command" key and the "Space" bar simultaneously to open Spotlight Search.

In the search field that appears, type "Terminal" and press "Return" or click on the "Terminal" application from the search results.

Once the Terminal is open, you will see a command-line interface where you can type commands and interact with the macOS command-line environment.

## Installing Python

Installing Python on a Mac is relatively straightforward. Here's a step-by-step guide to help you:

Check the installed version (optional): Open the Terminal application (found in the Utilities folder within the Applications folder) and type python --version to see if Python is already installed on your system. Note that macOS usually comes with a pre-installed version of Python.

Download Python: Visit the official Python website at https://www.python.org/downloads/ and click on the "Download Python" button. Choose the latest stable version suitable for your macOS.

Run the installer: Locate the downloaded Python installer package (e.g., python-3.x.x-macosx10.x.pkg) and double-click on it to start the installation process. Follow the prompts and instructions in the installer.

Customize installation (optional): During the installation, you'll have the option to customize the installation settings. If you're unsure, the default settings are usually sufficient for most users.

Install Python: Proceed with the installation by clicking "Install" or a similar button. You may be prompted to enter your administrator password. Wait for the installation to complete.

Verify the installation: After the installation is finished, open a new Terminal window and type python --version to verify that Python is installed correctly. You should see the version number of the installed Python.

## Installing pip

The pip package installer is usually bundled with Python by default. However, if you need to install or upgrade pip, you can follow these steps:

Open a terminal or command prompt on your system.

Check if you have Python installed by running the command python --version or python3 --version. This will display the installed Python version. If Python is not installed, you need to install it before proceeding.

Once you have confirmed that Python is installed, you can proceed to install pip using the following command:

```
python -m ensurepip --upgrade
```

Note: If you have multiple versions of Python installed, you may need to use python3 instead of python in the above command.

After running the command, pip should be installed or upgraded to the latest version.

To verify the installation, you can run pip --version or pip3 --version to check if pip is installed correctly and display its version.

That's it! You should now have pip installed on your system.

## Installing Docker - this step is optional

To install Docker on macOS, you can follow these steps:

Visit the Docker website to download the Docker Desktop for Mac installer: https://www.docker.com/products/docker-desktop

On the Docker Desktop for Mac page, click on the "Download Docker Desktop" button. You may be asked to create a Docker account or log in if you don't have one.

Once the download is complete, open the downloaded file (usually named Docker.dmg).

In the installer window, drag the Docker icon to the Applications folder to install Docker.

Open the Applications folder and double-click on the Docker icon to launch Docker.

You may be prompted to authorize the Docker app to run on your Mac. Enter your system password and follow the instructions to complete the authorization process.

Docker will start initializing, and you'll see the Docker icon appear in the macOS menu bar.

After Docker has finished starting up, you'll see the Docker whale icon in the menu bar. Docker is now installed and running on your Mac.

To verify that Docker is installed correctly, open a terminal and run the following command: docker version

This command will display the version information of Docker if it is installed properly.

You can now start using Docker on your macOS machine to build, run, and manage containers for your applications.

## Step 1 - Installing dependencies for the models (#step1)

After installing the software above, you will need to install the dependencies.  From the terminal, please run the commands below

```
pip install llama_index
pip install sentence_transformers
pip install transformers
pip install langchain
```

## Build your python script, T5pat.py

After installing the dependences, please build your python script.   In your terminal or code editor, please create a file, t5pat.py, in your directory i.e. t5pat, and cut and paste in following code into your t5pat.py file.

```
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
# service_context = ServiceContext.from_defaults(embed_model=embed_model, llm_predictor=llm_predictor) - original code line
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
```

Note - We found that the code, which is mostly from this github [issue](https://github.com/hwchase17/LangChain/issues/4438), would not run without the modification to the service_content statement.   We have left the original code line as a comment.   The modificiation is that we removed the embed_model reference, which was generating a failure message.   This parameter appears not be required for these models and removing it enables the program to run successfully.

For reference here is the error message with the original code
```
Traceback (most recent call last):
  File "/Users/xxxx/t5pat/t5pat.py", line 32, in <module>
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm_predictor=llm_predictor)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/xxxx/t5/.env/lib/python3.11/site-packages/llama_index/indices/service_context.py", line 147, in from_defaults
    embed_model.callback_manager = callback_manager
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pydantic/main.py", line 357, in pydantic.main.BaseModel.__setattr__
ValueError: "HuggingFaceEmbeddings" object has no field "callback_manager"
````

which we traced to this program and found this NOTE that states "the embed_model is not needed for all indices".   Based on this note, we removed it from the original call and the code executed to completion.

```
/Users/xxxx/t5/.env/lib/python3.11/site-packages/llama_index/indices/service_context.py

         # NOTE: the embed_model isn't used in all indices
146         embed_model = embed_model or OpenAIEmbedding()
147         embed_model.callback_manager = callback_manager
```

## Run your script

To run your script, please open your terminal to the directory and to the directory that holds the file.   Then run the following statement:

```
python t5pat.py
````

## Sample script output

The following provides sample model output from running the script:

```
Knowledge Retrieval Examples
==============================
Question: What is the capital of Germany?
Answer: berlin

Question: What is the capital of Spain?
Answer: turin

Question: What is the capital of Canada?
Answer: toronto


Question Answer Example, Text-to-Text
==============================
Question: The center of Tropical Storm Arlene, at 02/1800 UTC, is near 26.7N 86.2W. This position is about 425 km/230 nm to the west of Fort Myers in Florida, and it is about 550 km/297 nm to the NNW of the western tip of Cuba. The tropical storm is moving southward, or 175 degrees, 4 knots. The estimated minimum central pressure is 1002 mb. The maximum sustained wind speeds are 35 knots with gusts to 45 knots. The sea heights that are close to the tropical storm are ranging from 6 feet to a maximum of 10 feet.  Precipitation: scattered to numerous moderate is within 180 nm of the center in the NE quadrant.  Isolated moderate is from 25N to 27N between 80W and 84W, including parts of south Florida.  Broad surface low pressure extends from the area of the tropical storm, through the Yucatan Channel, into the NW part of the Caribbean Sea.   Where and when will the storm make landfall?
Answer: about 425 km/230 nm to the west of Fort Myers in Florida, and it is about 550 km/297 nm to the NNW of the western tip of Cuba


Question Answer with Reasoning Examples
==============================
Question: Logical Reasoning: What is the next number in the sequence: 2, 4, 6, 8, ...? If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?
Answer: Fluffy is a cat. Cats have tails. The next number in the sequence is 2. The answer: yes.

Question: Cause and Effect: If you eat too much junk food, what will happen to your health? How does smoking affect the risk of lung cancer?
Answer: no

Question: Analogical Reasoning: In the same way that "pen" is related to "paper", what is "fork" related to? If "tree" is related to "forest", what is "brick" related to?
Answer: bricks

Question: Deductive Reasoning: All dogs have fur. Max is a dog. Does Max have fur? If it is raining outside, and Mary doesn't like to get wet, will Mary take an umbrella?
Answer: Mary is a dog. An umbrella is used to keep people dry. Therefore, the final answer is yes.

Question: Inductive Reasoning: Every time John eats peanuts, he gets a rash. Does John have a peanut allergy? Every time Sarah studies for a test, she gets an A. Will Sarah get an A on the next test if she studies?
Answer: Sarah studies for a test every time she gets an A. Sarah will get an A on the next test if she studies.

Question: Counterfactual Reasoning: If I had studied harder, would I have passed the exam? What would have happened if Thomas Edison hadn't invented the light bulb
Answer: I would have passed the exam.
```

## Highlevel overview of the script

The script executes the following functions by the FLAN-T5-Large and all-miniLM-L6-v2 models:

For FLAN-T5-Large model:

1. AutoTokenizer.from_pretrained(model_id): Loads the tokenizer for the FLAN-T5-Large model.
2. AutoModelForSeq2SeqLM.from_pretrained(model_id): Loads the FLAN-T5-Large model for sequence-to-sequence language generation tasks.
3. pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512): Creates a pipeline object for text generation using the FLAN-T5-Large model.

For all-miniLM-L6-v2 model:

SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"): Loads the SentenceTransformer model with the all-miniLM-L6-v2 architecture for generating embeddings.

Additional components:

1. SimpleDirectoryReader(directory_path_).load_data(): Reads and loads data from a specified directory.
2. LLMPredictor(llm=local_llm_g_flan_t5_large): Creates an LLMPredictor object using the FLAN-T5-Large model for predicting outputs.
3. ServiceContext.from_defaults(llm_predictor=llm_predictor): Creates a ServiceContext object with default settings, including the LLMPredictor.
4. GPTListIndex.from_documents(documents, service_context=service_context): Creates an index using the GPTListIndex model based on the provided documents and service context.

The code performs text generation using the FLAN-T5-Large model and creates an index using the GPTListIndex model with the help of LLMPredictor and SentenceTransformerEmbeddings. The generated outputs and indexing status are printed for verification purposes.   The verification includes prompting the models to answer questions on knowledge retreival, text2text question answering and question answering with various forms of reasoning.

## Detailed review of the code blocks

The following provides a review of the code blocks:

### Importing the necessary dependencies:

```
from llama_index import LLMPredictor, ServiceContext, SimpleDirectoryReader, GPTListIndex
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.embeddings import SentenceTransformerEmbeddings
```

### Load models and pipelines

```
model_id = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
```

Here, the code specifies the model_id as 'google/flan-t5-large'. It then initializes the tokenizer and model using the AutoTokenizer and AutoModelForSeq2SeqLM classes from the Transformers library. The tokenizer is responsible for converting text into tokens that the model can process, while the model is a T5-based sequence-to-sequence language model.

```
pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=512
)
```

The code creates a text generation pipeline using the pipeline function from the Transformers library. The pipeline is initialized with the "text2text-generation" task, which indicates that the model will be used for generating text. The model and tokenizer are passed to the pipeline, along with a maximum sequence length of 512 tokens.

``` 
local_llm_g_flan_t5_large = HuggingFacePipeline(pipeline=pipe) 
```
Here, a HuggingFacePipeline object is created, wrapping the previously defined pipeline. This allows for convenient usage of the pipeline with additional functionalities.

### Test Knowledge retrieval

```
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
```

The code demonstrates the usage of the pipeline by generating text for different prompts for knowledge retreival. The pipeline takes a prompt as input and generates a text output based on the T5 model's trained capabilities.


### Indexing

```
directory_path = '/content/dir'  # Replace with the actual directory path
documents = SimpleDirectoryReader(directory_path).load_data()
```
The code specifies the directory_path_ variable as the path to a directory containing documents. The SimpleDirectoryReader is used to load the documents from the specified directory.

### Creating an embedding model:

``` 
embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") 
```

The code creates a SentenceTransformerEmbeddings object, which is initialized with the specified model name. This embedding model is used to convert text into vector representations.

### Creating an LLMPredictor object:

```
 llm_predictor = LLMPredictor(llm=local_llm_g_flan_t5_large) 
 ```

Here, an LLMPredictor object is created, which is initialized with the local_llm_g_flan_t5_large model. The LLMPredictor is responsible for making predictions using the provided language model.

### Creating a service context:

``` 
# service_context = ServiceContext.from_defaults(embed_model=embed_model, llm_predictor=llm_predictor) - original code line
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor) 
```

Here, a ServiceContext object is created using the from_defaults method. The llm_predictor parameter is set to the previously created llm_predictor object.   As mentioned above, this line commented out is the original line of code that was generating an error.

### Creating an index for document retrieval:

``` 
index = GPTListIndex.from_documents(documents, service_context=service_context) 
```

The code creates a GPTListIndex object using the from_documents method. The documents variable containing the loaded documents and the service_context are provided as arguments to the index creation.

## Text2Text Testing of Question and Answer

```
print("\nQuestion Answer Example, Text-to-Text")
print("=" * 30)

question1 = 'The center of Tropical Storm Arlene, at 02/1800 UTC, is near 26.7N 86.2W. This position is about 425 km/230 nm to the west of Fort Myers in Florida, and it is about 550 km/297 nm to the NNW of the western tip of Cuba. The tropical storm is moving southward, or 175 degrees, 4 knots. The estimated minimum central pressure is 1002 mb. The maximum sustained wind speeds are 35 knots with gusts to 45 knots. The sea heights that are close to the tropical storm are ranging from 6 feet to a maximum of 10 feet.  Precipitation: scattered to numerous moderate is within 180 nm of the center in the NE quadrant.  Isolated moderate is from 25N to 27N between 80W and 84W, including parts of south Florida.  Broad surface low pressure extends from the area of the tropical storm, through the Yucatan Channel, into the NW part of the Caribbean Sea.   Where and when will the storm make landfall?'

answer = local_llm_g_flan_t5_large(question1)
print(f"Question: {question1}")
print(f"Answer: {answer}\n")
```

This code prompts the model with a text2text question to answer.   It assigns the text of the question to question1 and then calls local_llm_g_flan_t5_large with question1 as it variable and assigns the response the variable called answer.

## Questions to test a variety of reasoning types

```
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
```
These questions test various forms of reasoning.   They use the same process.  Assign the text of the question to a question variable and use that variable to generate a response from the model.   We then print out the question and the answer.

## Review of the script's output

The following table provide summary of the model's answers.

| Task | Result |
| --- | --- |
| Knowledge retrieval | correct |
| Knowledge retrieval | incorrect |
| Knowledge retrieval | incorrect |
| Question Answer, Text2Text | incorrect |
| Logical Reasoning | incorrect |
| Logical Reasoning | correct |
| Analogical Reasoning | incorrect |
| Deductive Reasoning | incorrect |
| Deductive Reasoning | correct |
| Inductive Reasoning | correct |
| Counterfactual Reasoning | correct |


For our detailed review of the answers, let’s first examine the results of the flan_t5_large model for knowledge retreival.  

## Knowledge Retreival Example - output review

```
Knowledge Retrieval Examples
==============================
Question: What is the capital of Germany?
Answer: berlin

Question: What is the capital of Spain?
Answer: turin

Question: What is the capital of Canada?
Answer: toronto
```

The model provided answers to three questions on the capitals of Germany, Spain and Canada.  Generated answers: The lines berlin, turin, and toronto represent the generated answers for the given input prompts: "What is the capital of Germany?", "What is the capital of Spain?", and "What is the capital of Canada?" respectively. These answers are produced by the local_llm_g_flan_t5_large model used in the HuggingFacePipeline.

The model answered 2 of the 3 questions incorrectly.  When reviewing the incorrect answers, the model did provide cities in the correct country, just not the capital.

Berlin (correct)

Turin (wrong, it's Madrid)

Toronto (wrong, it's Ottowa)

## Question Answer Example - output review

```
Question Answer Example, Text-to-Text
==============================
Question: The center of Tropical Storm Arlene, at 02/1800 UTC, is near 26.7N 86.2W. This position is about 425 km/230 nm to the west of Fort Myers in Florida, and it is about 550 km/297 nm to the NNW of the western tip of Cuba. The tropical storm is moving southward, or 175 degrees, 4 knots. The estimated minimum central pressure is 1002 mb. The maximum sustained wind speeds are 35 knots with gusts to 45 knots. The sea heights that are close to the tropical storm are ranging from 6 feet to a maximum of 10 feet.  Precipitation: scattered to numerous moderate is within 180 nm of the center in the NE quadrant.  Isolated moderate is from 25N to 27N between 80W and 84W, including parts of south Florida.  Broad surface low pressure extends from the area of the tropical storm, through the Yucatan Channel, into the NW part of the Caribbean Sea.   Where and when will the storm make landfall?
Answer: about 425 km/230 nm to the west of Fort Myers in Florida, and it is about 550 km/297 nm to the NNW of the western tip of Cuba
```
The model answered this question inccorectly.   The model did answer with the location and time provided of the location defined in the text but it appeared not to understand the question with respect to landfall.


## Question Answer with Reasoning Examples - output review

The following provides analysis of the output of the Question Answer with Reasoning examples:

```
Question Answer with Reasoning Examples
==============================
Question: Logical Reasoning: What is the next number in the sequence: 2, 4, 6, 8, ...? If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?
Answer: Fluffy is a cat. Cats have tails. The next number in the sequence is 2. The answer: yes.
```

This question tests logical reasoning.  The model's answer for the sequence is incorrect and its answer for the cat is correct.

```
Question: Cause and Effect: If you eat too much junk food, what will happen to your health? How does smoking affect the risk of lung cancer?
Answer: no
```
This question tests cause and effort.  The model's answer is incorrect.

```
Question: Analogical Reasoning: In the same way that "pen" is related to "paper", what is "fork" related to? If "tree" is related to "forest", what is "brick" related to?
Answer: bricks
```
This question tests analogical reasoing.   The model's answer is incorrect.

```
Question: Deductive Reasoning: All dogs have fur. Max is a dog. Does Max have fur? If it is raining outside, and Mary doesn't like to get wet, will Mary take an umbrella?
Answer: Mary is a dog. An umbrella is used to keep people dry. Therefore, the final answer is yes.
```
This question tests deductive reasoning.   The part of the model's answer is incorrect and part of the model's answer is correct.

```
Question: Inductive Reasoning: Every time John eats peanuts, he gets a rash. Does John have a peanut allergy? Every time Sarah studies for a test, she gets an A. Will Sarah get an A on the next test if she studies?
Answer: Sarah studies for a test every time she gets an A. Sarah will get an A on the next test if she studies.
```
This question tests inductive reasoning.  The model's answer is correct.

```
Question: Counterfactual Reasoning: If I had studied harder, would I have passed the exam? What would have happened if Thomas Edison hadn't invented the light bulb
Answer: I would have passed the exam.
```
This question tests counterfactual reasoning.  The model's answer is correct.

As you can see, the model's performance can vary depending on the question type.   This is to be expected. 


## Running this script in a Docker container

This script can be run in a docker container.   Running a model in a container can have benefits, especially for easier portability and integration with Kubernetes clusters and model serving software like KServe.  Assuming you have docker installed, you can build a docker container using the code below, which can be created in a file named Dockerfile in your working directory i.e. t5pat.

Dockerfile

```
FROM python:3.9-slim
# Set the working directory inside the container

WORKDIR /app
# Copy the code file into the container

COPY t5pat.py .
# Install the required dependencies

RUN pip install --upgrade pip
RUN pip install torch torchvision
RUN pip install transformers
RUN pip install llama_index
RUN pip install sentence_transformers

# Set the command to run when the container starts
CMD ["python", "t5pat.py"]
```

## Build the docker container

To build your docker container, please run this command.   This could take 2-5 minutes to complete and will produce several status messages.

``` docker build -t t5pat . ```

## Run the docker container

To run your docker container, please run this command.  

``` docker run -it t5pat ```

Please be aware that your container should run similar to running the python script.   You may need to be patient while the software packages are being downloaded.

## Background links

GitHub issue on running LangChain locally
[https://github.com/hwchase17/LangChain/issues/4438](https://github.com/hwchase17/langchain/issues/4438)

Youtube walkthrough of running models locally
[https://www.youtube.com/watch?v=Kn7SX2Mx_Jk](https://www.youtube.com/watch?v=Kn7SX2Mx_Jk) 

What are embeddings
[https://vickiboykis.com/what_are_embeddings/](https://vickiboykis.com/what_are_embeddings/)

Previous Patterson Consulting post on using Huggingface with Docker and KServe.
[http://www.pattersonconsultingtn.com/blog/deploying_huggingface_with_kfserving.html](http://www.pattersonconsultingtn.com/blog/deploying_huggingface_with_kfserving.html) 
