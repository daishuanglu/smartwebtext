import os
import json
import pandas as pd
from typing import List
from tqdm import tqdm
import dataclasses

from lmapis import LLMCLient, sonnet, llama, oai
from utils.prnews import websearch
from utils import string_utils


OUTPUT_DIR = 'D://mlprojs/smartwebtext/newsdata/knowledge_graph'
INPUT_DIR = 'D://mlprojs/smartwebtext/newsdata/news'
LIMIT = None


def extract_pairs(lm: LLMCLient, list_of_search_files: List[str], output_dir: str):
    for fpath in tqdm(list_of_search_files, desc=f'run {lm.model_name}'):
        fbase = os.path.splitext(os.path.basename(fpath))[0]
        pairs = [[]]
        for il, line in enumerate(open(fpath, 'r')):
            if il > 0:
                news_article = websearch.load_body(line)
                results = lm(news_article)
                pairs.append(results)
        output_path = os.path.join(output_dir, lm.model_name, fbase + '.json')
        with open(output_path, 'w') as fp:
            json.dump(pairs, fp)


@dataclasses
class Extraction:
    company: str
    concept: str
    src_fid: str
    src_line_no: int


def load_extractions(pairs_dir):
    extractions = []
    for fname in os.listdir(pairs_dir):
        if fname.endswith('.json'):
            fpath = os.path.join(pairs_dir, fname)
            with open(fpath, 'r') as fp:
                data = json.load(fp)
                for i, lp in enumerate(data):
                    if lp:
                        sw = string_utils.stemmer.stem(lp['action'])
                        extraction = Extraction(lp['company'], sw, fname, i)
                        extractions.append(extraction)
    return extractions


def build_concepts(pairs_dir: str):
    """
    Each extraction results in a tuple associated with source file info.
    (concept, company) - file info: fid, line_no.
    """
    extractions: List[Extraction] = load_extractions(pairs_dir)
    df = pd.DataFrame()
    for ext in extractions:
        if ext.company not in df.index:
            df.loc[ext.company] = 0
        if ext.concept not in df.columns:
            df[ext.concept] = 0
        df.at[ext.company, ext.concept] += 1
    df.index.name = 'company'
    return df


if __name__ == '__main__':
    
    llm_clients = [
        sonnet.SonnetClient(), 
        llama.LlamaClient(),
        oai.OpenaiClient
        ]
    scrapped_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
                      if f.endswith('.txt')]
    if LIMIT is not None:
        scrapped_files = scrapped_files[:LIMIT]
    for lmc in llm_clients:
        extract_pairs(lmc, scrapped_files, OUTPUT_DIR)
    for lmc in llm_clients:
        pdir = os.path.join(OUTPUT_DIR, lmc.model_name)
        print(f'+ build concept for {pdir}.')
        df_concepts = build_concepts(pdir)
        concept_csv = os.path.join(OUTPUT_DIR, lmc.model_name + '_concept.csv')
        df_concepts.to_csv(concept_csv)