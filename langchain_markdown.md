

<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 0; ALERTS: 2.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>
<a href="#gdcalert2">alert2</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>



## 30 minutes to running LLMs on LangChain in a local environment

 

Within (30) minutes of reading this post, you should be able to complete model serving requests from two popular python-based large language models (LLM) using LangChain on your local computer without requiring the connection or costs to an external 3rd party API server, such as HuggingFaceHub or OpenAI.  


## Why run local

Some of the reasons why you may need to run your model locally, and not use an external API server, include::

* Security
    * You might want to fine tune the model and not post the derivative model on an external API server
* Cost
    * You might want to avoid paying an external company for API calls 
* Performance
    * You might be able to manage your model response times by using a private network and/or a specific server / processor type
* Functionality
    * Your model might only run locally (i.e. blenderbot)

This project provides the code and process to run two types of pretrained, large language models (FLAN-T5-Large and Sentence-BERT, all-MiniLM-L6-v2) using LangChain on your local computer. We selected these top performing models because several developers were having trouble running a tutorial locally, as tracked in this github issue, [https://github.com/hwchase17/LangChain/issues/4438](https://github.com/hwchase17/langchain/issues/4438).  

## LLM1 - Flan-t5-large

First, we will show the flan-t5-large model, which has 780M parameters and provides good performance for text-to-text and text-generation requirements as defined in this chart. It is a fairly popular model which had 446,125 downloads last month. For more detailed information on this model’s background, performance and capabilities, please see this link on HuggingFaceHub, [https://huggingface.co/google/flan-t5-large](https://huggingface.co/google/flan-t5-large).  

<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

![alt_text](image1.png "image_tooltip")

## LLM2 - S BERT

Second we will show a sentence-transformer model, specifically the BertModel  model_name='all-MiniLM-L6-v2'.  This model maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.  This model has been extensively evaluated for embedded sentences (Performance Sentence Embeddings) and for embedded search queries & paragraphs (Performance Semantic Search). 

The model is a general purpose model and was trained with more than 1 billion training pairs. The **all-MiniLM-L6-v2** is relatively small (80MB) and fast, yet it still offers good quality.  It is a very popular model and had 2,674,926 downloads last month.  The text below is from this page, [https://www.sbert.net/docs/pretrained_models.html](https://www.sbert.net/docs/pretrained_models.html). For more detailed information on this model’s background, performance and capabilities, please see this link [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).  

### SBERT.net, Sentence-Transformers - Pretrained Models

We provide various pre-trained models. Using these models is easy:

**from** **sentence_transformers** **import** SentenceTransformer

model = SentenceTransformer('model_name')

All models are hosted on the [HuggingFace Model Hub](https://huggingface.co/sentence-transformers).

### Model Overview

The following table provides an overview of (selected) models. They have been extensively evaluated for their quality to embedded sentences (Performance Sentence Embeddings) and to embedded search queries & paragraphs (Performance Semantic Search).

The **all-*** models where trained on all available training data (more than 1 billion training pairs) and are designed as **general purpose** models. The **all-mpnet-base-v2** model provides the best quality, while **all-MiniLM-L6-v2** is 5 times faster and still offers good quality. Toggle _All models_ to see all evaluated models or visit [HuggingFace Model Hub](https://huggingface.co/models?library=sentence-transformers) to view all existing sentence-transformers models.

<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image2.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

![alt_text](image2.png "image_tooltip")

## LangChain - What is it? Why use it?

The text in this section is from [https://python.LangChain.com/en/latest/index.html](https://python.langchain.com/en/latest/index.html) 

LangChain is a framework for developing applications powered by language models. We believe that the most powerful and differentiated applications will not only call out to a language model, but will also be:

1. _Data-aware_: connect a language model to other sources of data
2. _Agentic_: allow a language model to interact with its environment

The LangChain framework is designed around these principles.  This is the Python specific portion of the documentation. For a purely conceptual guide to LangChain, see [here](https://docs.langchain.com/docs/). For the JavaScript documentation, see [here](https://js.langchain.com/docs/). For concepts and terminology, please see [here](https://python.langchain.com/en/latest/getting_started/concepts.html).

### Modules

These modules are the core abstractions which we view as the building blocks of any LLM-powered application. For each module LangChain provides standard, extendable interfaces. LangChain also provides external integrations and even end-to-end implementations for off-the-shelf use. The docs for each module contain quickstart examples, how-to guides, reference docs, and conceptual guides.

The modules are (from least to most complex):

* [Models](https://python.langchain.com/en/latest/modules/models.html): Supported model types and integrations.
* [Prompts](https://python.langchain.com/en/latest/modules/prompts.html): Prompt management, optimization, and serialization.
* [Memory](https://python.langchain.com/en/latest/modules/memory.html): Memory refers to the state that is persisted between calls of a chain/agent.
* [Indexes](https://python.langchain.com/en/latest/modules/indexes.html): Language models become much more powerful when combined with application-specific data - this module contains interfaces and integrations for loading, querying and updating external data.
* [Chains](https://python.langchain.com/en/latest/modules/chains.html): Chains are structured sequences of calls (to an LLM or to a different utility).
* [Agents](https://python.langchain.com/en/latest/modules/agents.html): An agent is a Chain in which an LLM, given a high-level directive and a set of tools, repeatedly decides an action, executes the action and observes the outcome until the high-level directive is complete.
* [Callbacks](https://python.langchain.com/en/latest/modules/callbacks/getting_started.html): Callbacks let you log and stream the intermediate steps of any chain, making it easy to observe, debug, and evaluate the internals of an application.

### Use Cases

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

## Getting started

In our example and process, we wanted to simplify the getting started.   We selected specific LLMs to run in the LangChain framework which will run in a local environment i.e. in an older, Mac laptop.   We anticipate that many developers can use this initially and then modify our choices for your requirements.   

As you can see from the previous section, LangChain includes many advanced features and it enables complex model processing.   In our example, we will use models, prompts, and pipelines for question answering, text-to-text, sequence-to-sequence, and text-generation.

### Starting place

This post assumes that users have docker, python and a terminal installed and the installation for that software can be found on the linkes below.

### Install dependencies

After installing the software above, you will need to install the dependencies.  From the terminal, please run the commands below

```
pip install llama_index
pip install sentence_transformers
pip install transformers
pip install langchain
```

What about pytorch ?


### Build your python script, T5pat.py

After installing the dependences, please build your python script.   In your terminal or code editor, please create a file, T5pat.py, and copy in following code into it.

```
from llama_index import LLMPredictor, PromptHelper, ServiceContext, SimpleDirectoryReader, GPTVectorStoreIndex, GPTListIndex
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import pipeline
from LangChain.llms import HuggingFacePipeline 

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

from llama_index import GPTListIndex, SimpleDirectoryReader, GPTVectorStoreIndex
from LangChain.embeddings import SentenceTransformerEmbeddings 
from llama_index import LangChainEmbedding, ServiceContext

directory_path_ = '/content/dir'
documents = SimpleDirectoryReader(directory_path_).load_data()

llm_predictor = LLMPredictor(llm=local_llm_g_flan_t5_large)
embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# service_context = ServiceContext.from_defaults(embed_model=embed_model, llm_predictor=llm_predictor)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
index = GPTListIndex.from_documents(documents, service_context=service_context)
print("Indexing completed successfully")
print("llm_predictor", llm_predictor, "embed_model", embed_model, "service_content", service_context)

```

Note - We found that the code would not run without the modification to the service_content.   We have left the original code in place as a comment.   The modificiation is that we removed the embed_model reference, which was generating a failure message.   This parameter appears not be required for these models and removing it enables the program to run successfully.

### Output

The following provides the output from running the script:

```

berlin

turin

toronto

Indexing completed successfully

llm_predictor &lt;llama_index.llm_predictor.base.LLMPredictor object at 0x120b8b310> embed_model client=SentenceTransformer(

  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 

  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})

  (2): Normalize()

) model_name='all-MiniLM-L6-v2' cache_folder=None model_kwargs={} encode_kwargs={} service_content ServiceContext(llm_predictor=&lt;llama_index.llm_predictor.base.LLMPredictor object at 0x120b8b310>, prompt_helper=&lt;llama_index.indices.prompt_helper.PromptHelper object at 0x1245f3cd0>, embed_model=&lt;llama_index.embeddings.openai.OpenAIEmbedding object at 0x120f0d750>, node_parser=&lt;llama_index.node_parser.simple.SimpleNodeParser object at 0x121006f50>, llama_logger=&lt;llama_index.logger.base.LlamaLogger object at 0x1245f3c90>, callback_manager=&lt;llama_index.callbacks.base.CallbackManager object at 0x120bda850>)

```

## Review of the code

The following provides a review of the code blocks:

### Importing the necessary dependencies:

```
from llama_index import LLMPredictor, PromptHelper, ServiceContext, SimpleDirectoryReader, GPTVectorStoreIndex, GPTListIndex
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import pipeline
from LangChain.llms import HuggingFacePipeline
```

These lines import various modules and classes required for the code.

### Initializing the tokenizer and model:

```
model_id = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
```

Here, the code specifies the model_id as 'google/flan-t5-large'. It then initializes the tokenizer and model using the AutoTokenizer and AutoModelForSeq2SeqLM classes from the Transformers library. The tokenizer is responsible for converting text into tokens that the model can process, while the model is a T5-based sequence-to-sequence language model.

### Creating a text generation pipeline:

```
pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=512
)
```

The code creates a text generation pipeline using the pipeline function from the Transformers library. The pipeline is initialized with the "text2text-generation" task, which indicates that the model will be used for generating text. The model and tokenizer are passed to the pipeline, along with a maximum sequence length of 512 tokens.

### Creating a HuggingFacePipeline wrapper:

``` local_llm_g_flan_t5_large = HuggingFacePipeline(pipeline=pipe) ```

Here, a HuggingFacePipeline object is created, wrapping the previously defined pipeline. This allows for convenient usage of the pipeline with additional functionalities.

### Generating text using the pipeline:

```
print(local_llm_g_flan_t5_large('What is the capital of Germany? '))
print(local_llm_g_flan_t5_large('What is the capital of Spain? '))
print(local_llm_g_flan_t5_large('What is the capital of Canada? '))
```

The code demonstrates the usage of the pipeline by generating text for different prompts. The pipeline takes a prompt as input and generates a text output based on the T5 model's trained capabilities.

### Importing the necessary dependencies:

```
from llama_index import GPTListIndex, SimpleDirectoryReader, GPTVectorStoreIndex
from LangChain.embeddings import SentenceTransformerEmbeddings 
from llama_index import LangChainEmbedding, ServiceContext
```

These lines import the required modules and classes for indexing and embedding.

### Setting the directory path and loading documents:

```
directory_path_ = '/content/dir'
documents = SimpleDirectoryReader(directory_path_).load_data()
```

The code specifies the directory_path_ variable as the path to a directory containing documents. The SimpleDirectoryReader is used to load the documents from the specified directory.

### Creating an LLMPredictor object:

```
 llm_predictor = LLMPredictor(llm=local_llm_g_flan_t5_large) 
 ```

Here, an LLMPredictor object is created, which is initialized with the local_llm_g_flan_t5_large model. The LLMPredictor is responsible for making predictions using the provided language model.

### Creating an embedding model:

``` 
embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") 
```

The code creates a SentenceTransformerEmbeddings object, which is initialized with the specified model name. This embedding model is used to convert text into vector representations.

### Creating a service context:

``` 
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor) 
```

Here, a ServiceContext object is created using the from_defaults method. The llm_predictor parameter is set to the previously created llm_predictor object.

### Creating an index for document retrieval:

``` 
index = GPTListIndex.from_documents(documents, service_context=service_context) 
```

The code creates a GPTListIndex object using the from_documents method. The documents variable containing the loaded documents and the service_context are provided as arguments to the index creation.

### Printing indexing completion and service context information:

``` 
print("Indexing completed successfully")
print("llm_predictor", llm_predictor, "embed_model", embed_model, "service_content", service_context)
```

These lines simply print out a success message indicating the completion of the indexing process. It also prints information about the llm_predictor, embed_model, and service_context objects for verification purposes.


## Review of the output

1st let’s examine the results of the flan_t5_large model.   This model provided answers to three questions on the capitals of Germany, Spain and Canada.  It got 2 of the 3 answers wrong, but on the plus side, it did provide cities in the correct country, just not the capital.

Berlin (correct)

Turin (Madrid)

Toronto (Ottowa)

2nd let’s examine the results of the 'all-MiniLM-L6-v2 model. 

The output of the code snippets provides the following information:

Generated answers: The lines berlin, turin, and toronto represent the generated answers for the given input prompts: "What is the capital of Germany?", "What is the capital of Spain?", and "What is the capital of Canada?" respectively. These answers are produced by the local_llm_g_flan_t5_large model used in the HuggingFacePipeline.

Indexing completion: The line "Indexing completed successfully" indicates that the process of indexing the documents from the specified directory (directory_path_) was completed without any errors.

Service context information: The line beginning with "llm_predictor" provides information about the llm_predictor, embed_model, and service_context objects. It shows the object references (&lt;llama_index.llm_predictor.base.LLMPredictor object at 0x120b8b310>, client=SentenceTransformer(...), ServiceContext(...)) along with their configurations and parameters.

Overall, the output demonstrates the successful generation of answers, completion of indexing, and displays relevant information about the created objects and their configurations.


## Running this model in a Docker container

This code can be run in a docker container.   You can build a docker container by installing docker and then building a dockerfile using the code below.

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

Build the docker container

``` docker build -t t5pat . ```

Run the docker container

``` docker run -it t5pat ```


## Background links

Original post

[http://www.pattersonconsultingtn.com/blog/deploying_huggingface_with_kfserving.html](http://www.pattersonconsultingtn.com/blog/deploying_huggingface_with_kfserving.html) 

Issue running locally

[https://github.com/hwchase17/LangChain/issues/4438](https://github.com/hwchase17/langchain/issues/4438)

T5 model info

[https://huggingface.co/google/flan-t5-large](https://huggingface.co/google/flan-t5-large) 

Youtube walkthrough of running models locally

[https://www.youtube.com/watch?v=Kn7SX2Mx_Jk](https://www.youtube.com/watch?v=Kn7SX2Mx_Jk) 

What are embeddings

[https://vickiboykis.com/what_are_embeddings/](https://vickiboykis.com/what_are_embeddings/)

Falcon 7B, [https://huggingface.co/tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b) 


### Falcon 

Falcon-7B

Falcon-7B is a 7B parameters causal decoder-only model built by TII and trained on 1,500B tokens of RefinedWeb enhanced with curated corpora. It is made available under the Apache 2.0 license.

Downloads last month: 100,867


#### **Model Description**



* Developed by: [https://www.tii.ae](https://www.tii.ae/);
* Model type: Causal decoder-only;
* Language(s) (NLP): English and French;
* License: Apache 2.0.


### Why use Falcon-7B?



* It outperforms comparable open-source models (e.g., [MPT-7B](https://huggingface.co/mosaicml/mpt-7b), [StableLM](https://github.com/Stability-AI/StableLM), [RedPajama](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1) etc.), thanks to being trained on 1,500B tokens of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) enhanced with curated corpora. See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).
* It features an architecture optimized for inference, with FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) and multiquery ([Shazeer et al., 2019](https://arxiv.org/abs/1911.02150)).
* It is made available under a permissive Apache 2.0 license allowing for commercial use, without any royalties or restrictions.

⚠️ This is a raw, pretrained model, which should be further fine tuned for most use cases. If you are looking for a version better suited to taking generic instructions in a chat format, we recommend taking a look at [Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct).

Dependencies

pip install einops

pip install accelerate

