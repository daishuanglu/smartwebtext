import os
import json
import pandas as pd
from typing import List
from tqdm import tqdm
import dataclasses

from lmapis import LLMCLient, sonnet, llama, oai
from utils.prnews import websearch
from utils import string_utils


OUTPUT_DIR = '/mnt/d/mlprojs/smartwebtext/newsdata/knowledge_graph'
INPUT_DIR = '/mnt/d/mlprojs/smartwebtext/newsdata/news'
#OUTPUT_DIR = 'D://mlprojs/smartwebtext/newsdata/knowledge_graph'
#INPUT_DIR = 'D://mlprojs/smartwebtext/newsdata/news'
LIMIT = 34


def extract_pairs(lm: LLMCLient, list_of_search_files: List[str], output_dir: str):
    os.makedirs(os.path.join(OUTPUT_DIR, lm.model_name), exist_ok=True)
    nfiles = len(list_of_search_files)
    for i, fpath in enumerate(list_of_search_files):
        fbase = os.path.splitext(os.path.basename(fpath))[0]
        pairs = [[]]
        news_articles = list(websearch.load_key(fpath, 'Body'))
        for article in tqdm(news_articles, desc=f'run {lm.model_name} {i+1}/{nfiles}'):
            results = lm(article)
            pairs.append(results)
        output_path = os.path.join(output_dir, lm.model_name, fbase + '.json')
        with open(output_path, 'w') as fp:
            json.dump(pairs, fp)


@dataclasses.dataclass
class Extraction:
    company: str
    concept: str
    src_fid: str
    src_line_no: int


def load_extractions(pairs_dir, stem=True):
    extractions = []
    for fname in os.listdir(pairs_dir):
        if fname.endswith('.json'):
            fpath = os.path.join(pairs_dir, fname)
            with open(fpath, 'r') as fp:
                data = json.load(fp)
                for i, lps in enumerate(data):
                    if lps:
                        for lp in lps:
                            sw = lp['action']
                            if stem:
                                sw = ' '.join([
                                    string_utils.stemmer.stem(w) for w in lp['action'].split()])
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
    for ext in tqdm(extractions):
        if ext.concept not in df.columns:
            df[ext.concept] = 0
        if ext.company not in df.index:
            df.at[ext.company, ext.concept] = 0
        df.at[ext.company, ext.concept] += 1
    df.index.name = 'company'
    df = df.fillna(0.0)
    return df


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    llm_clients = [
        sonnet.SonnetClient(), 
        llama.LlamaClient(),
        oai.OpenaiClient(),
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