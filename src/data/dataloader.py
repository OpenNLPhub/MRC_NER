'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-08-25 17:49:20
 * @desc 
'''
import os
import torch
from torch.utils.data import Dataset,DataLoader,RandomSampler,SequentialSampler
from src.data.preprocess import MRCDataProcessor,FlatDataProcessor,HBTDataProcesser
import src.config.args as args
import src.config.ModelConfig as ModelConfig
from src.config.TrainingConfig import BertMRCTrainingConfig,HBTTrainingConfig
from src.utils.common import overrides,flatten_lists
from transformers import BertTokenizer,BertConfig



def create_Bert_tokenizer(use_pretrained=True,**kwargs):
    if use_pretrained:
        if 'model' not in kwargs:
            raise ValueError("Need type to sign the pretrained model")
        Path=ModelConfig.Bert_Pretrained_Model_Map.get(kwargs['model'],None)
        tokenizer_path=os.path.join(Path,'vocab.txt')
        if Path==None:
            raise ValueError("Need choice model again")
        Tokenizer=BertTokenizer(tokenizer_path)
    else:
        if 'vocab_file' not in kwargs:
            raise ValueError("Please input vocab file path")
        path=kwargs.get('vocab_file')
        Tokenizer=BertTokenizer(path)
    
    return Tokenizer

'''--------------------MRC DataSet--------------------- '''
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

def collate_fn(batch):
    return zip(*batch)

class FlatBertDataSet(Dataset):
    pass




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

    inputs,tags=processor.convert_units_to_features(units,args.MRC_TAG,tokenizer)

    MRC_DataSet=MRCBertDataSet(inputs,tags)

    MRC_DataLoader=DataLoader(MRC_DataSet,batch_size=BertMRCTrainingConfig.batch_size,shuffle=True,collate_fn=MRC_collate_fn)
    
    #注意 这个DataLoader 出来的Data 不是Tensor 还需要自己将其变为Tensor
    return MRC_DataLoader

def create_Flat_DataLoader(mode,data_dir,use_pretrained=True):
    pass
    



'''--------------HBT-----------------'''

class HBTDataSet(Dataset):
    def __init__(self,**kwargs):
        self.input_idx=kwargs['input_idx']
        self.attention_mask=kwargs['attention_mask']
        self.chosen_sub_idx_list=kwargs['chosen_sub_idx']
        self.sub_start_vec_list=kwargs['sub_start_vec']
        self.sub_end_vec_list=kwargs['sub_end_vec']
        self.obj_start_vec_list=kwargs['obj_start_vec']
        self.obj_end_vec_list=kwargs['obj_end_vec']

    def __getitem__(self,index):
        return self.input_idx[index],self.attention_mask[index],self.chosen_sub_idx_list[index]\
            ,self.sub_start_vec_list[index],self.sub_end_vec_list[index]\
            ,self.obj_start_vec_list[index],self.obj_start_vec_list[index]

    def __len__(self):
        return len(self.input_idx)


def create_HBT_DataLoader(mode,data_dir,tokenizer):
    assert mode in ['train','dev','test']
    processor=HBTDataProcesser()
    if mode=='train':
        text,subject_lists,triple_list=processor.get_train_units(data_dir)
    elif mode=='dev':
        text,subject_lists,triple_list=processor.get_dev_units(data_dir)
    else:
        text,subject_lists,triple_list=processor.get_test_units(data_dir)
    relation_list=processor.get_labels(data_dir)
    
    data=processor.convert_units_to_features(text,triple_list,subject_lists,relation_list,tokenizer)
    # import pdb
    # pdb.set_trace()
    dataset=HBTDataSet(**data)
    dataloader=DataLoader(dataset,batch_size=HBTTrainingConfig.batch_size,shuffle=True,collate_fn=collate_fn)

    return dataloader


def test_dataloader(data_dir,tokenizer,create_dataloader):
    dataloader=create_dataloader('train',data_dir,tokenizer)
    print('test finish')
    for data in dataloader:
        import pdb
        pdb.set_trace()



    
if __name__=='__main__':
    path=os.path.join(args.HBT_SOURCE_DATA,'triple')
    test_dataloader(path,create_Bert_tokenizer(True,model='bert-base-uncased'),create_HBT_DataLoader)
