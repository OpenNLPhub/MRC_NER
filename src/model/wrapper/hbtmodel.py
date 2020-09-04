
import torch
import torch.nn as nn
import torch.optim as optim
from src.config.TrainingConfig import HBTTrainingConfig
from src.model.wrapper.BaseWrapper import BaseWrapper
from src.model.nn.bert import create_bert_encoder
from src.model.nn.hbt import HBT
from src.utils.common import overrides
import numpy as np
from src.metrics.metrics import confusion_matrix_to_units

class hbtModel(BaseWrapper):
    def __init__(self,relation_list,bert):
        super(hbtModel,self).__init__()
        self.relation_list=relation_list
        self.model=HBT(bert,len(relation_list))
        
        self.model.to(self.device)

        '''
        重新自定义训练参数
        '''
        self.lr=HBTTrainingConfig.lr
        self.epoches=HBTTrainingConfig.epoches
        self.batch_size=HBTTrainingConfig.batch_size

        self.optimizer=optim.Adam(self.model.parameters(),lr=self.lr)
    

    def _trans_data2tensor(self,batch_data):
        input_ids,attention_mask,chosen_sub_idx,sub_start_vec,sub_end_vec,\
            obj_start_vec,obj_end_vec=batch_data
        F=lambda x:torch.from_numpy(np.array(x)).long().to(self.device)
        input_ids=F(input_ids)
        attention_mask=F(attention_mask)
        sub_start_vec=F(sub_start_vec)
        sub_end_vec=F(sub_end_vec)
        obj_start_vec=F(obj_start_vec)
        obj_end_vec=F(obj_end_vec)

        #这里利用attention_mask 将所有target的padding部分除去
        #计算loss，evaluate的时候都会将predict的结果faltten，要与pred size 保持一致
        FF=lambda x,mask: x.masked_select(mask)
        mask=attention_mask==1
        sub_start_vec=FF(sub_start_vec,mask)
        sub_end_vec=FF(sub_end_vec,mask)
    
        mask=mask.squeeze(2).expand(-1,-1,obj_start_vec.shape[-1])
        obj_start_vec=FF(obj_start_vec,mask)
        obj_end_vec=FF(obj_end_vec,mask)

        #处理chosen_sub_idx
        batch_size,max_seq_len,word_emb_size=input_ids.shape

        sub_start_ids,sub_end_ids=zip(*chosen_sub_idx)
        
        sub_start_ids=F(sub_start_ids)
        sub_end_ids=F(sub_end_ids)
        # [batch_size, *]
        sub_start_ids=sub_start_ids.unsqueeze(1).expand(-1,max_seq_len).unsqueeze(2).expand(-1,-1,word_emb_size)
        sub_end_ids=sub_end_ids.unsqueeze(1).expand(-1,max_seq_len).unsqueeze(2).expand(-1,-1,word_emb_size)

        return input_ids,attention_mask,(sub_start_ids,sub_end_ids),\
            sub_start_vec,sub_end_vec,obj_start_vec,obj_end_vec
    
    
    @overrides(BaseWrapper)
    def _cal_loss(self,batch_data,**kwargs):
        input_ids,attention_mask,chosen_sub,sub_start_vec,sub_end_vec,\
            obj_start_vec,obj_end_vec=self._trans_data2tensor(batch_data)
        
        pred_sub_start,pred_sub_end,pred_obj_start,pred_obj_end=self.model(input_ids=input_ids,attention_mask=attention_mask,chosen_sub=chosen_sub)

        loss=self.model.cal_loss((pred_sub_start,pred_sub_end),(sub_start_vec,sub_end_vec),\
            (pred_obj_start,pred_obj_end),(obj_start_vec,obj_end_vec))
        
        return loss

    @overrides(BaseWrapper)
    def _eval_unit(self,batch_data,**kwargs):
        threshold=kwargs.get('threshold',0.5)
        #threshold default to 0.5
        
        input_ids,attention_mask,chosen_sub,sub_start_vec,sub_end_vec,\
            obj_start_vec,obj_end_vec=self._trans_data2tensor(batch_data)

        pred_sub_start,pred_sub_end,pred_obj_start,pred_obj_end=self.model.predict(input_ids=input_ids,attention_mask=attention_mask,\
            chosen_sub=chosen_sub,threshold=threshold)

        EVAL=lambda y,y_,id2label:confusion_matrix_to_units(y,y_,id2label,binary=True)
        units=[\
            EVAL(sub_start_vec,pred_sub_start,{0:1,1:'subject_start'}),\
            EVAL(sub_end_vec,pred_sub_end,{0:1,1:'subject_end'}),\
            EVAL(obj_start_vec,pred_obj_start,{0:1,1:'object_start'}),\
            EVAL(obj_end_vec,pred_obj_end,{0:1,1:'object_end'})
        ]
        return units
        

        