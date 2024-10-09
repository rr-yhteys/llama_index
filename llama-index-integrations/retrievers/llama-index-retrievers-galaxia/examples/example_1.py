#! /usr/bin/env python3

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.retrievers.galaxia import GalaxiaRetriever
import json

path_json = 'credentials.json'
with open(path_json) as f_tmp:
    smbb_api_key = json.load(f_tmp)["smbb_api_key"]

smbb_api_url = "https://dev.api.smabbler.com"
smbb_knowledge_base_id = "<fillin>"

gr = GalaxiaRetriever(
    smbb_api_url,
    smbb_api_key,
    smbb_knowledge_base_id,
)

query = "What is Marie Curie's nationality?"

result = gr.retrieve(query)

print(result)

breakpoint()
