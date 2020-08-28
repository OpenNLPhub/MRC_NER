'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-08-28 14:13:41
 * @desc 
'''

import os
from src.data.dataloader import create_MRC_DataLoader,create_Bert_tokenizer
from src.model.wrapper.MRCBert_NER import MRCBert_NER
import src.config.args as args
from src.metrics.metrics import convert_units_to_dataframe
from src.utils.common import save_model,load_model

def run_Bert_MRC_NER():

    model_is_existed=os.path.exists(args.MRC_MODEL_PATH)
    tokenizer=create_Bert_tokenizer(True)
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
