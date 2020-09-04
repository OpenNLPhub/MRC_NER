from transformers import BertModel,BertConfig
from src.config.ModelConfig import Bert_Pretrained_Model_Map


def create_bert_encoder(use_pretrained=True,**kwargs):
    if use_pretrained:
        model=kwargs.get('model')
        if model==None:
            raise ValueError('need model parameter')
        modelpath=Bert_Pretrained_Model_Map.get(model)
        if modelpath==None:
            raise ValueError('mistake input in model')
        bertconfig=BertConfig.from_pretrained(modelpath,author='lingze')
        bertmodel=BertModel.from_pretrained(modelpath,config=bertconfig)
    else:
        bertconfig=BertConfig(**kwargs,author='lingze')
        bertmodel=BertModel(config=bertconfig)
    
    return bertmodel
        
    
