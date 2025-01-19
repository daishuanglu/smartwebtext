import json
import datetime
import os
import blake3
import traceback
from dotenv import load_dotenv

load_dotenv()


class LLMCLient():

    def __init__(self, model: str = ''):
        self._cache_dir = os.getenv('LLM_CACHE_DIR')
        os.makedirs(self._cache_dir, exist_ok=True)
        self.model = model

    @property
    def model_name(self):
        raise NotImplementedError('.model_name property must be created at children of LLMCLient.')

    def _hash_request_id(self, request: str):
        ustr = self.model + '-' + str(request)
        h = blake3.blake3(ustr.encode('utf-8'))
        cid =  h.hexdigest()
        return cid

    def _cache(self, request, response, contents):
        cid = self._hash_request_id(request)
        ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        cache_path = os.path.join(self._cache_dir, cid)
        with open(cache_path, 'w') as fp:
            json.dump({'request': request,
                       'response': response,
                       'contents': contents,
                       'req_id': f'{self.model}_{ts}'}, fp)

    def try_load_contents_from_cache(self, request: str):
        cached_fids = {f for f in os.listdir(self._cache_dir)}
        sid = self._hash_request_id(request)
        if sid in cached_fids:
            with open(os.path.join(self._cache_dir, sid), 'r') as fp:
                message = json.load(fp)
                return message['contents']

    def isvalid(self, line: str):
        if "here is" in line.lower():
            return False
        if "here are" in line.lower():
            return False
        if "Note" in line:
            return False
        if "i can extract" in line.lower():
            return False
        return True 

    def parse_content_to_json(self, raw_response: str):
        resp_lines = raw_response.split('\n')
        resp_lines = [line for line in resp_lines if self.isvalid(line)]
        raw_response = '\n'.join(resp_lines)
        try:
            output = json.loads(raw_response)
            return output
        except Exception as e:
            print(e)
            print('raw llm response: ', raw_response)

    def make_request(self):
        raise NotImplementedError(
            "The LLM make request function must be instantiated in the child client.")

    def __call__(self, prompt_string_or_path: str):
        prp_string = prompt_string_or_path
        if os.path.exists(prompt_string_or_path):
            with open(prompt_string_or_path) as fp:
                prp_string = fp.read()
        try:
            contents = self.try_load_contents_from_cache(prp_string)
            if contents is None:
                contents, resp = self.make_request(prp_string)
                self._cache(prp_string, resp, contents)
            results = []
            for content in contents:
                json_dicts = self.parse_content_to_json(content)
                if json_dicts is not None:
                    results.extend(json_dicts)
            return results
        except Exception as e:
            print(e)
            print(traceback.print_exc())
            return []