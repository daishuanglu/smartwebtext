import pandas as pd
from typing import Dict, List

from utils import string_utils
from preprocessors import dataloader


PRNEWS_FILEPATTERN = 'data_model/scrapped_news/webtext_thread_*.txt'
PRNEWS_SEPARATOR = '\t'
PRNEWS_HEADERS = ['Company', 'Stock', 'date', 'Title', 'Body']
PRNEWS_INVALID_KEYTERMS = ['follow us', 'twitter|', 'linkedin|', '.copyright', 'www.',
    'facebook:', 'visit:', 'twitter:', 'for more information', 'click here', 'email', 'phone', 'logo']
PRNEWS_SENTENCE_MIN_WORDS = 5
PRNEWS_PARAGRAPH_SEP = ';;;;'
PRNEWS_DATA_SEP = '\t'
PRNEWS_MUST_CONTAIN_COLS = ['Text', 'Company']
PRNEWS_EVAL_DIR = 'evaluation/prnews_accounting'
FASTTEXT_HOME = 'fasttext'


def prnews_text_preproc(s):
    #stopwords = string_utils.load_stopwords()
    s = s.lower()
    s = [w.strip() for w in s.split(PRNEWS_PARAGRAPH_SEP)]
    s = sum([string_utils.split_to_sentences(ss) for ss in s], [])
    for term in PRNEWS_INVALID_KEYTERMS:
        s = list(filter(lambda x: term not in x, s))
    s = [string_utils.remove_punct(sp) for sp in s]
    s = list(filter(lambda x: len(x.split()) > PRNEWS_SENTENCE_MIN_WORDS, s))
    s = [string_utils.clean_char(sp).strip() for sp in s]
    #s = [string_utils.remove_stopwords(sp, stopwords, sub=' ').strip() for sp in s]
    s = [string_utils.replace_consecutive_spaces(sp) for sp in s]
    return PRNEWS_PARAGRAPH_SEP.join(s).strip()


def prnews(
        output_files, split_ratio, vocabs: Dict[str, List[str]] = {}):

    FASTTEXT_TOOL = string_utils.fasttext_toolkits(fasttext_model_home=FASTTEXT_HOME)

    def _body(s):
        if not FASTTEXT_TOOL.detEN([s.lower()])[0]:
            return ''
        return prnews_text_preproc(s)

    def _title(s):
        s = string_utils.get_title_name(s, sep='-').strip()
        s = s if FASTTEXT_TOOL.detEN([s])[0] else ''
        return s

    def _company(s):
        return string_utils.getcompanyname(s, sep='-').strip()

    def _text(row):
        t, b = row['Title'].lower(), row['Body'].lower()
        if t == '' and b == '':
            return ''
        else:
            s = PRNEWS_PARAGRAPH_SEP.join(
                [row['Title'].lower(), row['Body'].lower()]).strip()
            company_name = string_utils.remove_company_suffix(row['Company'].lower()).strip()
            return s.replace(company_name, '')

    textfile = dataloader.BaseTextFile(
        fpath=PRNEWS_FILEPATTERN, sep=PRNEWS_SEPARATOR,
        types={'Body': _body, 'Company': _company, 'Title': _title},
        col_fns=[('Text', _text)], must_contain=PRNEWS_MUST_CONTAIN_COLS)
    textfile.write(
        output_files=output_files,
        split_ratio=split_ratio,
        cols=['Company', 'Title', 'Text'],
        sep=PRNEWS_DATA_SEP)
    for vocab_path, vocab_cols in vocabs.items():
        textfile.vocab(vocab_cols, vocab_path)


def prnews_concept_examples(fpath, text_col, ref_col, sep):
    df = pd.read_csv(
        fpath, sep=sep, dtype=str, parse_dates=False, na_values=[], keep_default_na=False)

    def _clean_row(r):
        s = prnews_text_preproc(str(r[text_col]).lower())
        c = string_utils.remove_company_suffix(str(r[ref_col]).lower())
        s = s.replace(c, '')
        return s

    paragraphs = df.apply(_clean_row, axis=1).tolist()
    ss = []
    for sentences in paragraphs:
        ss += sentences.strip().split(PRNEWS_PARAGRAPH_SEP)
    return ss