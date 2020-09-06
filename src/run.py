'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-08-28 14:13:41
 * @desc 
'''

import os
from src.data.dataloader import create_MRC_DataLoader,create_Bert_tokenizer,\
    create_HBT_DataLoader
from src.model.wrapper.MRCBert_NER import MRCBert_NER
from src.model.wrapper.HBT import hbtModel
import src.config.args as args
from src.metrics.metrics import convert_units_to_dataframe
from src.utils.common import save_model,load_model


def run_Bert_MRC_NER():

    model_is_existed=os.path.exists(args.MRC_MODEL_PATH)
    tokenizer=create_Bert_tokenizer(True,model='bert-base-chinese')
    labels=args.MRC_TAG
    if not model_is_existed:
        train_data_loader=create_MRC_DataLoader('train',args.MRC_SOURCE_DATA,tokenizer)
        dev_data_loader=create_MRC_DataLoader('dev',args.MRC_SOURCE_DATA,tokenizer)
        vocab_size=len(tokenizer.__dict__.get('vocab'))
        model=MRCBert_NER(len(labels),True)
        model.train(train_data_loader,dev_data_loader,tokenizer=tokenizer,label_class=labels)
        save_model(model,args.MRC_MODEL_PATH)
    else:
        model=load_model(args.MRC_MODEL_PATH)
    test_data_loader=create_MRC_DataLoader('test',args.MRC_SOURCE_DATA,tokenizer)
    units=model.test(test_data_loader,tokenizer=tokenizer,label_class=args.MRC_TAG)
    df=convert_units_to_dataframe(units)
    df.to_csv(args.MRC_RESULT_PATH)

def run_HBT():
    model_is_existed=os.path.exists(args.HBT_MODEL_PATH)
    path=os.path.join(args.HBT_SOURCE_DATA,'triple')
    tokenizer=create_Bert_tokenizer(True,model='bert-base-uncased')
    if not model_is_existed:
        train_data_loader=create_HBT_DataLoader('train',path,tokenizer)
        dev_data_loader=create_HBT_DataLoader('dev',path,tokenizer)
        relation_list=[]
        with open(os.path.join(path,'relation_type.txt'),'r') as f:
            for line in f.readlines():
                relation_list.append(line.strip()) 
        model=hbtModel(len(relation_list),use_pretrained=True,model='vert-base-uncased')
        model.train(train_data_loader,dev_data_loader)
        save_model(model,args.HBT_MODEL_PATH)
    else:
        model=load_model(args.MRC_MODEL_PATH)
    test_data_loader=create_HBT_DataLoader('test',path,tokenizer)
    units=model.test(test_data_loader,tokenizer=tokenizer)
    df=convert_units_to_dataframe(units)
    df.to_csv(args.HBT_RESULT_PATH)

    



    
