import os
import json
import pandas as pd
import re
from typing import List

from bertopic import BERTopic
from utils.prnews import websearch
from utils import string_utils


STOPWORDS = string_utils.load_stopwords()


def load_company_docs(raw_data_dir):
    
    for f in os.listdir(raw_data_dir):
        cid = f.split('_')[-1].removesuffix('.txt')
        fpath = os.path.join(raw_data_dir, f)
        docs = [lines for lines in websearch.load_key(fpath, 'Body')]
        doc_lines = [line for lines in docs for line in lines.split('\n')]
        # Preprocessings
        doc_lines = [re.sub('[^a-z ]', '', line.lower()) for line in doc_lines]
        doc_lines = [string_utils.remove_stopwords(line, STOPWORDS) for line in doc_lines]
        yield int(cid), doc_lines


def load_concept_df_from_json_pairs(pair_json_files: List[str], cnames = None):
    df = pd.DataFrame()
    for json_file in pair_json_files:
        cid = int(os.path.splitext(os.path.basename(json_file))[0])
        with open(json_file, 'r') as fp:
            pairs = json.load(fp)
        for p in pairs:
            ct, cp = p['action'], p['company']
            if cnames is not None:
                cp = cnames[cid]
            if ct not in df.columns:
                df[ct] = 0
            if cp not in df.index:
                df.at[cp, ct] = 0
            df.at[cp, ct] += 1
    df.index.name = 'company'
    df = df.fillna(0)
    return df


COMP_URLS = 'newsdata/companyurls.txt'
KG_JSON_DIR = 'newsdata/knowledge_graph/bert'
CONCEPT_CSV_PATH = 'newsdata/knowledge_graph/bert_concept.csv'
LIMIT = 34


def main():
    config = {
        'raw_data_dir': 'newsdata/news',
        'model_name': 'bert',
    }
    cnames = [string_utils.getcompanyname(
        os.path.basename(c_url.strip('/\n'))) for c_url in open(COMP_URLS, 'r')]
    os.makedirs(KG_JSON_DIR, exist_ok=True)
    limit_doc = min(LIMIT, len(cnames)) if LIMIT is not None else len(cnames)
    json_pairs = []
    for id, (cid, docs) in enumerate(load_company_docs(config['raw_data_dir'])):
        print(f'+ Processing {id}/{limit_doc}, {len(docs)} text lines.')
        if len(docs) < 10:
            continue
        if id > limit_doc:
            break
        if os.path.exists(f'{KG_JSON_DIR}/{cid}.json'):
            json_pairs.append(f'{KG_JSON_DIR}/{cid}.json')
            continue
        cp = cnames[cid]
        pairs = []
        topic_model = BERTopic(verbose=True)
        topic_model.fit_transform(docs)
        df_topics = topic_model.get_topic_info()
        tnames = df_topics[df_topics['Topic'] > 0]['Name']
        print(tnames)
        if tnames.any():
            tnames = tnames.apply(lambda x: str(x).split('_')[1:])
            for tns in tnames:
                pairs.extend([{'company': cp, 'action': tn} for tn in tns if tn])
            json_pairs.append(f'{KG_JSON_DIR}/{cid}.json')
            with open(f'{KG_JSON_DIR}/{cid}.json', 'w') as fp:
                json.dump(pairs, fp)
    df = load_concept_df_from_json_pairs(json_pairs, cnames)
    df.to_csv(CONCEPT_CSV_PATH)


if __name__ == '__main__':
    main()
