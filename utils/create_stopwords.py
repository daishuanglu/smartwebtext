
from nltk.corpus import stopwords
import spacy
from gensim.parsing.preprocessing import STOPWORDS


cachedStopwords = sorted(set(list(stopwords.words("english"))
                       + list(STOPWORDS)
                       + list(spacy.load("en_core_web_sm").Defaults.stop_words)
                       ))
with open('utils/gensim_nltk_spacy_stopwords.txt', 'w') as f:
    for w in cachedStopwords:
        f.write(w + '\n')