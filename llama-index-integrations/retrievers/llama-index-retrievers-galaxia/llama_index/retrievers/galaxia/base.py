import time

from typing import List, Optional, Dict, Any

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

import smabbler
from smabbler.api.client.models.algorithm_versions_response_model import AlgorithmVersionsResponseModel



def analyse_smbb(text, alg_version, api_instance, n_retries=5):
    ''' send query to API, wait for the result, retrieve the result '''

    # init analysis
    initialize_operation_request_model = smabbler.api.client.InitializeOperationRequestModel()
    initialize_operation_request_model.algorithm_version = alg_version
    initialize_operation_request_model.text = text

    api_response_init = None 

    for n in range(n_retries):
        try:
            api_response_init = api_instance.analyze_initialize_post(
                initialize_operation_request_model)

        except Exception as e:
            print("Exception when calling DefaultApi->analyze_initialize_post: %s\n" % e)
            time.sleep(2)
            continue
        
        else:
            break

    if api_response_init is None:
        raise Exception("couldn't initialize operation")

    # check if results available and get them or timeout
    operation_status_model = smabbler.api.client.OperationStatusModel()
    operation_status_model.operation_id = api_response_init.operation_id

    for n in range(n_retries):
        # check if results are available
        try:
            api_response = api_instance.analyze_status_post(operation_status_model)
        except Exception as e:
            print("Exception when calling DefaultApi->analyze_status_post: %s\n" % e) 
            time.sleep(2)
            continue

        
        if api_response.status == 'processed':
            # get the results
            try:
                api_response = api_instance.analyze_result_post(operation_status_model)
                return api_response
                
            except Exception as e:
                print("Exception when calling DefaultApi->analyze_result_post: %s\n" % e)
                return
                
        time.sleep(2)

    #print('Timeout')
    return


def extract_results(res):
    ''' extracts annotations from smabbler output '''
    return [i.category for i in res.result.items]



class GalaxiaRetriever(BaseRetriever):
    """ Galaxia knowledge retriever

    See <link to our docs> for more info.

    Args:
        smbb_api_url : url of smabbler API, e.g. "https://beta.api.smabbler.com"
        smbb_api_key : API key
        smbb_knowledge_base_id : ID pf the knowledge base

    Example:
        .. code-block:: python

            from llama_index.retrievers.galaxia import GalaxiaRetriever

            retriever = ... 

    """
    def __init__(
        self, 
        smbb_api_url: str,
        smbb_api_key: str,
        smbb_knowledge_base_id: str,
        callback_manager: Optional[CallbackManager] = None,
        ):

        self.configuration = smabbler.api.client.Configuration(
            host=smbb_api_url
        )
        self.configuration.api_key['api_key'] = smbb_api_key
        self.knowledge_base_id = smbb_knowledge_base_id

        self.api_client = smabbler.api.client.ApiClient(self.configuration)

        super().__init__(callback_manager)


    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        ''' 

        '''
        query = query_bundle.query_str
        api_instance = smabbler.api.client.DefaultApi(self.api_client) 
        available_mods = api_instance.algorithm_versions_get()

        response = analyse_smbb(
            query, 
            self.knowledge_base_id, 
            api_instance, 
            n_retries=5
        )

        score = 1
        metadata = {}
        node_with_score = []
        for res in extract_results(response):
            node_with_score.append(
               NodeWithScore(
                   node=TextNode(
                       text=res,
                       metadata=metadata,
                   ),
                   score=score,
               )
           )
       

        return node_with_score


    def __del__(self):
        del self.api_client

