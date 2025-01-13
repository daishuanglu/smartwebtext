"""
Llama API (https://docs.llama-api.com/api-reference/endpoint/create) is a separate service hosting
Llama models. The company is not owned by Meta.
"""
import os
import json
import llamaapi
import lmapis


DEFAULT_RESP_TEMPLATE = [
    {'company': 'abc', 'action': 'analytics'},
    {'company': "dcb", 'action': "provided analytics platform"}]


class LlamaClient(lmapis.LLMCLient):
    
    def __init__(self,
                 model='llama3.3-70b', 
                 example_response = DEFAULT_RESP_TEMPLATE):
        super(LlamaClient, self).__init__(model)
        self.client = llamaapi.LlamaAPI(api_token=os.environ.get("LLAMA_API_KEY"))
        self.resp_ep = json.dumps(example_response)

    @property
    def model_name(self):
        return 'llama'

    def make_request(self, corpus: str):
        api_request_json = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"""
                    You are a business and accounting researcher extracting the business strategies
                    that companies mentioned they used in a press news annoucement. Use simple
                    keyword or phrases to represent a business strategy. Return your
                    extraction in a list of json dictionary. For example {self.resp_ep} Do not add
                    any exta explanations so I can load the list in python.
                    """
                },
                {
                    "role": "user",
                    "content": f"Here is the full text of the press news: {corpus}"
                },
                {
                    "role": "user",
                    "content": """
                    What are all the companies and their business strategies mentioned in this press
                    annoucement article?
                    """
                }
            ],
            "stream": False,
            }
        response = self.client.run(api_request_json)
        resp_dict = json.loads(response.content.decode('utf-8'))
        contents = [c['message']['content'] for c in resp_dict['choices']]
        return contents, resp_dict
    
