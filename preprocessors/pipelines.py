from utils import string_utils
from preprocessors import dataloader


PRNEWS_FILEPATTERN = 'data_model/scrapped_news/webtext_thread_*.txt'
PRNEWS_SEPARATOR = '\t'
PRNEWS_HEADERS = ['Company', 'Stock', 'date', 'Title', 'Body']
PRNEWS_INVALID_KEYTERMS = ['follow us', 'twitter|', 'linkedin|', '.copyright',
    'facebook:', 'visit:', 'twitter:', 'for more information', 'click here', 'email', 'phone', 'logo']
PRNEWS_SENTENCE_MIN_WORDS = 5
PRNEWS_PARAGRAPH_SEP = ';;;;'
PRNEWS_DATA_SEP = '\t'
PRNEWS_MUST_CONTAIN_COLS = ['Text', 'Company']

KTH_ACTION_SRC_CSV = 'data_model/kth_actions.csv'
KTH_ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
KTH_ACTION_DATA_CSV_SEP = ','
KTH_SPLIT_COL = 'split'
KTH_SPLIT_CSV = 'data_model/kth_actions_{split}.csv'

def prnews_text_preproc(s):
    stopwords = string_utils.load_stopwords()
    s = s.lower()
    s = [w.strip() for w in s.split(PRNEWS_PARAGRAPH_SEP)]
    for term in PRNEWS_INVALID_KEYTERMS:
        s = list(filter(lambda x: term not in x.lower(), s))
    s = [string_utils.remove_punct(sp) for sp in s]
    s = list(filter(lambda x: len(x.split()) > PRNEWS_SENTENCE_MIN_WORDS, s))
    s = [string_utils.clean_char(sp) for sp in s]
    s = [string_utils.remove_stopwords(sp, stopwords, sub=' ') for sp in s]
    return PRNEWS_PARAGRAPH_SEP.join(s).strip()


def prnews(output_files, split_ratio, vocab_path):

    def _body(s):
        return prnews_text_preproc(s)

    def _title(s):
        s = string_utils.get_title_name(s, sep='-').strip()
        return s

    def _company(s):
        return string_utils.getcompanyname(s, sep='-').strip()

    def _text(row):
        return PRNEWS_PARAGRAPH_SEP.join(
            [row['Title'].lower(), row['Body'].lower()]).strip()

    textfile = dataloader.BaseTextFile(
        fpath=PRNEWS_FILEPATTERN, sep=PRNEWS_SEPARATOR,
        types={'Body': _body, 'Company': _company, 'Title': _title},
        col_fns=[('Text', _text)], must_contain=PRNEWS_MUST_CONTAIN_COLS)
    textfile.write(
        output_files=output_files,
        split_ratio=split_ratio,
        cols=['Company', 'Text'],
        explode_sep=PRNEWS_PARAGRAPH_SEP,
        sep=PRNEWS_DATA_SEP)
    textfile.vocab(['Company'], vocab_path)
    return

def kth_action_video():
    textfile = dataloader.PandasTextFile(fpath=KTH_ACTION_SRC_CSV, sep=KTH_ACTION_DATA_CSV_SEP)
    df = textfile.all()
    for split in df[KTH_SPLIT_COL].unique():
        df[df[KTH_SPLIT_COL] == split].to_csv(KTH_SPLIT_CSV.format(split=split), index=False)
        print('KTH action ', split, 'set: ', len(df), 'samples')
    return

def eval_example_sentences(
        fpath, text_col, row_delimiter, multi_sent_seps, preproc_sep, preproc_fn=None):

    def _text_col(s):
        s = [w.strip() for w in string_utils.split_by_multi_sep(s, multi_sent_seps)]
        s = [w for w in s if w]
        if preproc_fn is None:
            return preproc_sep.join(s)
        else:
            return preproc_fn(preproc_sep.join(s))

    textfile = dataloader.PandasTextFile(fpath=fpath, sep=row_delimiter, types={text_col: _text_col})
    texts = textfile.all()[text_col]
    return texts.dropna().tolist()