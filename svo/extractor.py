
import spacy

nlp = spacy.load('en_core_web_sm')

def get_subject_phrase(doc):
    for token in doc:
        if ("subj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]


def get_object_phrase(doc):
    ind=[i for i,token in enumerate(doc) if 'obj' in token.dep_]
    if not ind:
        return
    return doc[ind[0]:ind[-1]+1]
    '''
        if ("dobj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]
    '''

def get_svo(sentence):
    doc = nlp(sentence)
    subject_phrase = get_subject_phrase(doc)
    object_phrase = get_object_phrase(doc)
    #print(subject_phrase)
    #print(object_phrase)
    return subject_phrase,object_phrase