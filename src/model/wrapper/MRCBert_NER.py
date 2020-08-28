import torch
import numpy as np
import torch.optim as optim
import src.config.args as args
from src.model.wrapper.BaseWrapper import BaseWrapper
from src.model.nn.MRCBert import MRCBert
from src.config.ModelConfig import BertConfig
from src.utils.common import overrides,flatten_lists
from src.metrics.metrics import Eval_Unit,confusion_matrix_to_units
from src.config.TrainingConfig import BertMRCTrainingConfig

class MRCBert_NER(BaseWrapper):
    def __init__(self,num_labels,use_pretrained=True):
        super(MRCBert_NER,self).__init__()
        if use_pretrained:
            self.model=MRCBert(num_labels=num_labels)
        else:
            self.model=MRCBert(num_labels=num_labels,vocab_size=BertConfig.vocab_size,\
                hidden_size=BertConfig.hidden_size,emb_size=BertConfig.emb_size,use_pretrained=False)

        self.model.to(self.device)
        '''
        可重写自定义 训练参数
        '''
        self.lr=BertMRCTrainingConfig.lr
        self.epoches=BertMRCTrainingConfig.epoches
        self.batch_size=BertMRCTrainingConfig.batch_size
        self.optimizer=optim.Adam(self.model.parameters(),lr=self.lr)
    

    def __trans_data2tensor(self,batch_data):
        input_ids,attention_mask,token_type_ids,tags_lists=batch_data
        # transform data to tensor
        input_ids=torch.from_numpy(np.array(input_ids)).long().to(self.device)
        attention_mask=torch.from_numpy(np.array(attention_mask)).long().to(self.device)
        token_type_ids=torch.from_numpy(np.array(token_type_ids)).long().to(self.device)
        tags=torch.from_numpy(np.array(flatten_lists(tags_lists))).to(self.device)
        return input_ids,attention_mask,token_type_ids,tags
    
    @overrides(BaseWrapper)
    def _cal_loss(self,batch_data,**kwargs):
        tokenizer=kwargs.get('tokenizer',None)
        labels=kwargs.get('label_class',None)
        labels2id={i:j for j,i in enumerate(labels)}
        if tokenizer==None or labels==None:
            raise ValueError('Need tokenizer and label_class')
        input_ids,attention_mask,token_type_ids,tags=self.__trans_data2tensor(batch_data)
        score=self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,\
            berttokenizer=tokenizer)
        # 将 'O' 的weight降低
        weight_ce=torch.ones(len(labels)).to(self.device)
        weight_ce[labels2id.get('O')]=0.05
        # import pdb; pdb.set_trace()
        loss=self.model.cal_loss(score,tags,weight_ce)
        return loss

    
    #该method 只用于评测模型的效果，不评测在特定任务下的效果
    #返回一个自定义的unit list
    @overrides(BaseWrapper)
    def test(self,test_data_loader,**kwargs):
        tokenizer=kwargs.get('tokenizer',None)
        labels=kwargs.get('label_class',None)
        if tokenizer==None or labels==None:
            raise ValueError('Need tokenizer and label_class')
        self.best_model.eval()
        ids2labels={(i,label) for i,label in enumerate(labels)}
        ans=None
        total_step=test_data_loader.dataset.__len__()//self.batch_size +1
        
        with torch.no_grad():
            for step,batch_data in enumerate(test_data_loader):
                input_ids,attention_mask,token_type_ids,tags=self.__trans_data2tensor(batch_data)
                pred=self.best_model.predict(input_ids=input_ids,attention_mask=attention_mask,\
                    token_type_ids=token_type_ids,berttokenizer=tokenizer)
                
                units=confusion_matrix_to_units(pred,tags,ids2labels)
                ans=[i+j for i,j in zip(ans,units)] if ans!=None else units
                if step % 10 ==0:
                    print('step/total_step:{}/{}'.format(step,total_step))
                    for unit in ans:
                        unit.ptr()
        return ans
                
        
                

