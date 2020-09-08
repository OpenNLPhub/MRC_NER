import json
import os
from tqdm import tqdm
import src.config.args as args
from src.data.preprocess import DataProcessor
from src.utils.common import overrides
from random import choice

def get_relation_type():
    pass

def check_units(t,data):
    length=[]
    label_type={}
    n=len(data)
    print('check {} data'.format(t))
    max_len=-1
    for i,d in tqdm(enumerate(data)):
        text=d['text']
        max_len=len(text.split()) if len(text.split())>max_len else max_len
        length.append(len(text.split()))
        for s,r,o in d['triple_list']:
            if label_type.get(r) == None:
                label_type[r]=1
    print('check {} data, {} items in data , max len sentence is {}\t the average text length:{}\tthe num of relation type {}'\
            .format(t,n,max_len,float(sum(length))/n,len(list(label_type.keys()))))
    return list(label_type.keys())

def check_dataset(dir:str):
    with open(os.path.join(dir,'train.json'),'r') as train_f,\
         open(os.path.join(dir,'test.json'),'r') as test_f,\
         open(os.path.join(dir,'dev.json'),'r') as dev_f:
         train_data=json.loads(train_f.read())
         dev_data=json.loads(dev_f.read())
         test_data=json.loads(test_f.read())
         all_data=[*train_data,*dev_data,*test_data]
         
         length=[]
         label_type={}

         train_label=check_units('train',train_data)
         dev_label=check_units('dev',dev_data)
         test_label=check_units('test',test_data)
         check_units('all',all_data)
         expect_label=set(dev_label+test_label)-set(train_label)
         expect_item=0
         expect_num=0
         for i,d in tqdm(enumerate(all_data)):
             text=d['text']
             for s,r,o in d['triple_list']:
                 f=True
                 if r in expect_label:
                     expect_num+=1
                     if f:
                         f=False
                         expect_item+=1
        
         print(expect_label)
         print('no existed relation : data item:{} relation mention num:{}'\
             .format(expect_item,expect_num))

    
def build_relation_list(dir:str):
    with open(os.path.join(dir,'train.json'),'r') as train_f:
        rel_type={}
        train_data=json.loads(train_f.read())
        for i,d in tqdm(enumerate(train_data)):
            for s,r,o in d['triple_list']:
                if rel_type.get(r) == None:
                    rel_type[r]=1
    
    rel_type=list(rel_type.keys())

    with open(os.path.join(dir,'relation_type.txt'),'w') as f:
        for rel in rel_type:
            f.write(rel+'\n')
    
    return rel_type




if __name__=='__main__':
    check_dataset(os.path.join(args.HBT_SOURCE_DATA,'triple'))
    build_relation_list(os.path.join(args.HBT_SOURCE_DATA,'triple'))