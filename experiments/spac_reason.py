import os
import json

from lmapis import llama
from utils import pdf_utils


RAW_PDF_DIR = "lawsuitdata/pdf"
SUMMARY_DIR = "lawsuitdata/summary"
OUTPUT_PDF_DIR = 'lawsuitdata/pdf_reason'
OUTPUT_SUMY_DIR = 'lawsuitdata/summary_reason'


if __name__ == '__main__':
    for odir in [OUTPUT_PDF_DIR, OUTPUT_SUMY_DIR]:
        os.makedirs(odir, exist_ok=True)
    llama_client = llama.LlamaLawsuitClient()
    for fname in os.listdir(RAW_PDF_DIR):
        cname = fname.removesuffix('.pdf')
        print('+', fname, ' pdf')
        fpath = os.path.join(RAW_PDF_DIR, fname)
        page_texts = pdf_utils.load_pdf_text(fpath)
        page_texts = page_texts[2:-5]
        list_of_reasons = llama_client('\n'.join(page_texts))
        output_path = os.path.join(OUTPUT_PDF_DIR, cname + '.json')
        with open(output_path, 'w') as fp:
            json.dump(list_of_reasons, fp, indent=4)
    for fname in os.listdir(SUMMARY_DIR):
        cname = fname.removesuffix('.txt')
        print('+', fname, ' summary')
        fpath = os.path.join(SUMMARY_DIR, fname)
        with open(fpath, 'r', encoding='utf-8') as fp:
            texts = fp.read()
        list_of_reasons = llama_client(texts)
        output_path = os.path.join(OUTPUT_SUMY_DIR, cname + '.json')
        with open(output_path, 'w') as fp:
            json.dump(list_of_reasons, fp, indent=4)
