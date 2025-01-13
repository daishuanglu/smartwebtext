"""
OpenAI API usage: https://platform.openai.com/settings/organization/billing/overview.
"""
import os
import json
import openai

import lmapis


DEFAULT_RESP_TEMPLATE = [
    {'company': 'abc', 'action': 'analytics'},
    {'company': "dcb", 'action': "provided analytics platform"}]


class OpenaiClient(lmapis.LLMCLient):
    
    def __init__(self,
                 model='gpt-4o-mini', 
                 example_response = DEFAULT_RESP_TEMPLATE):
        super(OpenaiClient, self).__init__(model)
        self.client = openai.OpenAI(
            organization=os.environ.get('OPENAI_ORG_ID'),
            api_key=os.environ.get("OPENAI_API_KEY"))
        self.resp_ep = json.dumps(example_response)

    @property
    def model_name(self):
        return 'openai'

    def make_request(self, corpus: str):
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[
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
            ]
        )
        return [c.message.content for c in response.choices], response.to_dict()
    
