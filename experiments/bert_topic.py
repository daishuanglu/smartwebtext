import os
import json
import pandas as pd

from bertopic import BERTopic
from utils.prnews import websearch
from utils import string_utils


def load_company_docs(raw_data_dir):
    
    for f in os.listdir(raw_data_dir):
        cid = f.split('_')[-1].removesuffix('.txt')
        fpath = os.path.join(raw_data_dir, f)
        docs = [lines for lines in websearch.load_key(fpath, 'Body')]
        yield int(cid), [line for lines in docs for line in lines.split('\n')] 


COMP_URLS = 'newsdata/companyurls.txt'
KG_JSON_DIR = 'newsdata/knowledge_graph/bert'
CONCEPT_CSV_PATH = 'newsdata/knowledge_graph/bert_concept.csv'


def main():
    config = {
        'raw_data_dir': 'newsdata/news',
        'model_name': 'bert',
    }
    cnames = [string_utils.getcompanyname(
        os.path.basename(c_url.strip())) for c_url in open(COMP_URLS, 'r')]
    df = pd.DataFrame()
    os.makedirs(KG_JSON_DIR, exist_ok=True)
    for cid, docs in load_company_docs(config['raw_data_dir']):
        print(f'+ Processing {cid}/{len(cnames)}, {len(docs)} text lines.')
        if len(docs) < 10:
            continue
        pairs = []
        cp = cnames[cid]
        topic_model = BERTopic(verbose=True)
        topic_model.fit_transform(docs)
        df_topics = topic_model.get_topic_info()
        tnames = df_topics[df_topics['Topic'] > 0]['Name']
        print(tnames)
        if tnames.any():
            tnames = tnames.apply(lambda x: str(x).split('_')[1:])
            for tns in tnames:
                pairs.extend([{'company': cp, 'action': tn} for tn in tns if tn])
            with open(f'{KG_JSON_DIR}/{cid}.json', 'w') as fp:
                json.dump(pairs, fp)
            for tns in tnames:
                for ct in tns:
                    if ct:
                        if ct not in df.columns:
                            df[ct] = 0
                        if cp not in df.index:
                            df.at[cp, ct] = 0
                        df.at[cp, ct] += 1
    df.index.name = 'company'
    df.to_csv(CONCEPT_CSV_PATH)


if __name__ == '__main__':
    main()
