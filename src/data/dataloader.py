'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-08-25 17:49:20
 * @desc 
'''
import torch
from torch.utils.data import Dataset,DataLoader,RandomSampler,SequentialSampler
from src.data.preprocess import MRCDataProcessor,FlatDataProcessor,convert_units_to_features
import src.config.args as args
import src.config.ModelConfig as ModelConfig
from src.config.TrainingConfig import BertMRCTrainingConfig
from src.utils.common import overrides,flatten_lists
from transformers import BertTokenizer,BertConfig

class MRCBertDataSet(Dataset):
    def __init__(self,inputs,tags_lists):
        super(MRCBertDataSet,self).__init__()
        # import pdb; pdb.set_trace()
        self.input_ids=inputs['input_ids']
        self.attention_mask=inputs['attention_mask']
        self.token_type_ids=inputs['token_type_ids']
        self.tags_lists=tags_lists

    def __getitem__(self,index):
        # data={'inputs':(self.input_ids[index],self.attention_mask[index],self.token_type_ids[index]),'tag':self.tags_lists[index]}
        return self.input_ids[index],self.attention_mask[index],self.token_type_ids[index],self.tags_lists[index]

    def __len__(self):
        # assert len(self.input_ids) == len(self.tags_lists)
        return len(self.input_ids)

def MRC_collate_fn(batch):
    input_ids,attention_mask,token_type_ids,tags_lists=zip(*batch)
    return input_ids,attention_mask,token_type_ids,tags_lists

class FlatBertDataSet(Dataset):
    pass

def create_Bert_tokenizer(use_pretrained=True):
    if use_pretrained:
        Tokenizer=BertTokenizer.from_pretrained(ModelConfig.BERT_BASE_CHINESE)
    else:
        Tokenizer=BertTokenizer(args.VOCAB_FILE)
    
    return Tokenizer

def create_MRC_DataLoader(mode,data_dir,tokenizer):
    assert mode in ['train','dev','test']
    processor=MRCDataProcessor()
    if mode =='train':
        units=processor.get_train_units(data_dir)
    elif mode=='dev':
        units=processor.get_dev_units(data_dir)
    else:
        units=processor.get_test_units(data_dir)
    
    # import pdb;pdb.set_trace()

    inputs,tags=convert_units_to_features(units,args.MRC_TAG,tokenizer)

    MRC_DataSet=MRCBertDataSet(inputs,tags)

    MRC_DataLoader=DataLoader(MRC_DataSet,batch_size=BertMRCTrainingConfig.batch_size,shuffle=True,collate_fn=MRC_collate_fn)
    
    #注意 这个DataLoader 出来的Data 不是Tensor 还需要自己将其变为Tensor
    return MRC_DataLoader

def create_Flat_DataLoader(mode,data_dir,use_pretrained=True):
    pass
    


    
if __name__=='__main__':
    data_dir=args.MRC_SOURCE_DATA
    tokenizer=create_Bert_tokenizer()
    trainLoader=create_MRC_DataLoader('train',data_dir,tokenizer)
    print('test finish')
    # for data in trainLoader:
    #     # import pdb
    #     # pdb.set_trace()
    #     inputs_ids,attention_mask,token_type_ids,tag_list=data
        