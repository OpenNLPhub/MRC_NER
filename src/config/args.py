import os
from os.path import join
cwd=os.getcwd()
''' Path config'''

RAW_SOURCE_DATA=join(cwd,'data','raw')
FLAT_SOURCE_DATA=join(cwd,'data','flat')
MRC_SOURCE_DATA=join(cwd,'data','mrc')

MRC_LABEL=['LOC','PER','ORG','O']
LABELS=['B-LOC','I-LOC','B-PER','I-LOC','B-ORG','I-ORG','O']

