import os
from os.path import join
cwd=os.getcwd()
''' Path config'''

BERT_BASE_CHINESE=join(cwd,'data','static','bert-base-chinese')


#可以在这里设定自己定义的BertConfig
class BertConfig():
    hidden_size=0
    emb_size=0
    vocab_size=0
    