import json
import os
from tqdm import tqdm
import src.config.args as args


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
        max_len=len(text) if len(text)>max_len else max_len
        length.append(len(text))
        for s,r,o in d['triple_list']:
            if label_type.get(r,True):
                label_type[r]=1
    print('check {} data, {} items in data , max len sentence is {}\the average text length:{}\tthe num of relation type {}'\
            .format(t,n,max_len,float(sum(length))/n,len(list(label_type.keys()))))
        
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

         check_units('train',train_data)
         check_units('dev',dev_data)
         check_units('test',test_data)
         check_units('all',all_data)

    



if __name__=='__main__':
    check_dataset(args.HBT_SOURCE_DATA)