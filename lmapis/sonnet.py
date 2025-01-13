"""
Anthropic Sonnet billing: https://console.anthropic.com/settings/cost. 
"""
import os
import json
import anthropic
import lmapis


DEFAULT_RESP_TEMPLATE = [
    {'company': 'abc', 'action': 'analytics'},
    {'company': "dcb", 'action': "provided analytics platform"}]


class SonnetClient(lmapis.LLMCLient):
    
    def __init__(self,
                 model='claude-3-5-sonnet-20241022', 
                 example_response = DEFAULT_RESP_TEMPLATE):
        super(SonnetClient, self).__init__(model)
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.resp_ep = json.dumps(example_response)

    @property
    def model_name(self):
        return 'sonnet'

    def make_request(self, corpus: str):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": f"""
                    You are a business and accounting researcher extracting the business strategies
                    that companies mentioned they used in a press news annoucement. Use simple
                    keyword or phrases to represent a business strategy. Return your
                    extraction in a list of json dictionary. For example {self.resp_ep} Do not add
                    any exta explanations so I can load the list in python.
                    """,
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "type": "text",
                    "text": f"Here is the full text of the press news: {corpus}",
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": """
                    What are all the companies and their business strategies mentioned in this press
                    annoucement article?
                    """,
                }
            ]
        )
        return [content.text for content in response.content], response.to_dict()
    
