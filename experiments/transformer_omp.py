import cv2
import torch
from transformers import BertModel, AutoTokenizer, DebertaTokenizerFast, DebertaModel
import numpy as np
import numpy.linalg as la
import torch.nn.functional as F
from utils import visual_utils

bert_model_name = 'bert-large-uncased'
model = BertModel.from_pretrained(bert_model_name)
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def sum_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1)


def text_repr(text, target=False):
    encoding = tokenizer([text],
                           return_tensors='pt',
                           padding="max_length",
                           truncation=True,
                           # add_special_tokens=True,
                           max_length=512)
    encoder_output = model(**encoding)
    toks = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    e = torch.sigmoid(encoder_output['last_hidden_state'])
    #e = encoder_output['last_hidden_state']
    e = F.normalize(e, p=2, dim=-1)
    if target:
        #attn_mask = encoding['attention_mask'].clone()
        #attn_mask[0, 0] = 0
        #attn_mask[0, encoding['attention_mask'].sum(1)-1] = 0
        e = mean_pooling(e, encoding['attention_mask'])
    return e, toks


def matching_pursuit(y: np.ndarray,
                     D: np.ndarray,
                     eps_min=1e-3,
                     iter_max=1000) -> np.ndarray:
    """

    Args:
        y: time_series_length. Target_vector for sparse representation.
        D: num_basis * time_series_length
        eps_min: eps
        iter_max: max iteration for MP.

    Returns:
        1 * num_basis An amplitude vector x, where y = D * x = x^T * D
    """
    s = np.zeros(D.shape[0], dtype=np.float64)
    r = y.astype(np.float64)
    for ii in range(iter_max):
        dot = np.dot(D, r)
        i = np.argmax(np.abs(dot))
        s[i] = dot[i]
        r -= s[i] * D[i, :]
        if la.norm(r) < eps_min:
            break
        print('iter %d pursuit loss=%.6f' % (ii, la.norm(r)))
    return np.atleast_2d(s)


def norm01(x):
    return (x - x.min())/(x.max() - x.min())


context = "I bought this for my husband who plays the piano. " \
       "He is having a wonderful time playing these old hymns.  " \
       "The music is at times hard to read because we think the book was " \
       "published for singing from more than playing from. Great purchase though!"

target = "Bought a piano book for husband, who's enjoying playing old hymns. " \
         "Sheet music is a bit challenging; may be more suited for singing. " \
         "Still, a great purchase!"

#context = "Today is May 23, 2023. I went to hiking at Mountain Rainier."
#target = "May 23"
#target = "hard to read"
#target = "a good purchase"
#target = "hard to read."

D, D_tokens = text_repr(context, target=False)
y, y_tokens = text_repr(target, target=True)
y, D = y.detach().cpu().numpy()[0], D.detach().cpu().numpy()[0]
x = matching_pursuit(y, D, eps_min=1e-3, iter_max=1000)
basis_amp_hm = (x.transpose() * D).mean(1)
print(D_tokens)
print(basis_amp_hm[:20])
sid = D_tokens.index('[CLS]') + 1
eid = D_tokens.index('[SEP]') - 1
viz = visual_utils.text_class_activation_map(D_tokens, norm01(basis_amp_hm), start_end_ids=(sid, eid))
#sparse_hm = basis_amp_hm[sid: eid]
#sparsity = (np.abs(sparse_hm) < 1e-5).sum()
#print('sparsity (number of zeros)=', sparsity, '/%d' % len(sparse_hm), f'={sparsity/(eid-sid)}',)
#viz = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
cv2.imwrite('context_heatmap.jpg', viz)


#  sentence transformer is not good.
"""
def sent_text_repr(text, target=False):
    encoding = tokenizer([text],
                         return_tensors='pt',
                         padding="max_length",
                         truncation=True,
                         # add_special_tokens=True,
                         max_length=512)
    encoder_output = model(encoding)
    e = encoder_output['token_embeddings']
    toks = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    if target:
        e = encoder_output['sentence_embedding']
    e = F.normalize(e, p=2, dim=-1)
    return e, toks
"""