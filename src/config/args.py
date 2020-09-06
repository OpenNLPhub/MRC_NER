import os
from os.path import join
cwd=os.getcwd()
''' Path config'''

RAW_SOURCE_DATA=join(cwd,'data','raw')
FLAT_SOURCE_DATA=join(cwd,'data','flat')
MRC_SOURCE_DATA=join(cwd,'data','mrc')

MRC_LABEL=['LOC','PER','ORG','O']
LABELS=['B-LOC','I-LOC','B-PER','I-LOC','B-ORG','I-ORG','O']
MRC_TAG=['B','I','O']

STOP_WORD_LIST=None


FLAG_WORDS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
VOCAB_FILE=join(RAW_SOURCE_DATA,'vocab.txt')



HBT_SOURCE_DATA=join(cwd,'data','WebNLG')


''' ----  model parameter path ----'''

MRC_MODEL_PATH=join(cwd,'data','model','mrc.pkl')
HBT_MODEL_PATH=join(cwd,'data','model','hbt.pkl')


'''  ---- output result path ---- '''
MRC_RESULT_PATH=join(cwd,'data','output','mrc.csv')
HBT_RESULT_PATH=join(cwd,'data','output','hbt.csv')