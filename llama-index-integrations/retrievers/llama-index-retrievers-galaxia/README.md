# LlamaIndex Retrievers Integration: Galaxia

## Galaxia Knowledge Base

> Galaxia Knowledge Base is an integrated knowledge base and retrieval mechanism for RAG. In contrast to standard solution, it is based on Knowledge Graphs built using symbolic NLP and Knowledge Representation solutions. Provided texts are analysed and transformed into Graphs containing text, language and semantic information. This rich structure provides advantages for RAG retrieval:

- it bypassess the need for chunking by retrieving elementary information extracted from texts
- the retrieval isn't based on vector distance, but semantic information

Implementing RAG using Galaxia involves first uploading your files to [Galaxia](). After processing, you can use `GalaxiaRetriever` to connect to the API and start retrieving.

## Installation

```
pip install llama-index-retrievers-galaxia
```

## Usage

```
from llama_index.retrievers.galaxia import GalaxiaRetriever

gr = GalaxiaRetriever(
    smbb_api_url,
    smbb_api_key,
    smbb_knowledge_base_id,
)

query = "What is Marie Curie's nationality?"

retrieved_results = gr.retrieve(query)

print(retrieved_results)

```

## Notebook

Explore the retriever using Notebook present at:
#TODO
