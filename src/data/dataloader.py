'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-08-25 17:49:20
 * @desc 
'''
import torch
from torch.utils.data import Dataset,DataLoader,RandomSampler,SequentialSampler

from preprocess import MRCDataProcessor,FlatDataProcessor,convert_units_to_features
import src.config.args as args
import src.config.ModelConfig as ModelConfig
from src.config.TrainingConfig import BertMRCTrainingConfig
from src.utils.common import overrides
from transformers import BertTokenizer,BertConfig
class MRCBertDataSet(Dataset):
    def __init__(self,inputs,tags_lists):
        super(MRCBertDataSet,self).__init__()
        self.input_ids=inputs['inputs_ids']
        self.attention_mask=inputs['attention_mask']
        self.token_type_ids=inputs['token_type_ids']
        self.tags_lists=tags_lists

    @overrides(Dataset)
    def __getitem__(self,index):

        return (self.input_ids[index],self.attention_mask[index],self.token_type_ids[index]),self.tags_lists[index]

    @overrides(Dataset)
    def __len__(self):
        # assert len(self.input_ids) == len(self.tags_lists)
        return len(self.input_ids)

class FlatBertDataSet(Dataset):
    pass


def create_MRC_DataLoader(mode,data_dir,use_pretrained=True):
    assert mode in ['train','dev','test']
    processor=MRCDataProcessor()
    if mode =='train':
        units=processor.get_train_units(data_dir)
    elif mode=='dev':
        units=processor.get_dev_units(data_dir)
    else:
        units=processor.get_test_units(data_dir)
    
    if use_pretrained:
        Tokenizer=BertTokenizer.from_pretrained(ModelConfig.BERT_BASE_CHINESE)
    else:
        Tokenizer=BertTokenizer(args.VOCAB_FILE)
    
    inputs,tags=convert_units_to_features(units,args.MRC_TAG,Tokenizer)

    MRC_DataSet=MRCBertDataSet(inputs,tags)

    MRC_DataLoader=DataLoader(MRC_DataSet,batch_size=BertMRCTrainingConfig.batch_size,shuffle=True,num_workers=2)
    
    #注意 这个DataLoader 出来的Data 不是Tensor 还需要自己将其变为Tensor
    return MRC_DataLoader

def create_Flat_DataLoader(mode,data_dir,use_pretrained=True):
    pass
    


    
