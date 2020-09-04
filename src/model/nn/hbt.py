
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class HBT(nn.Module):
    def __init__(self,bertmodel,relation_num):
        super(HBT,self).__init__()
        self.bert=bertmodel
        self.word_embed_size=bertmodel.config.hidden_size
        self.subject_tagger=Tagger(self.word_emb_size,torch.sigmoid,1)
        self.object_tagger=Tagger(self.word_emb_size,torch.sigmoid,relation_num)
        self.loss_layer=nn.BCELoss()
        self.relation_nun=relation_num

    def forward(self,input_ids,attention_mask,chosen_sub):
        score=self.bert(input_ids=input_ids,attention_mask=attention_mask)
        #batch_size * max_seq_len * word_emb_size
        
        pred_sub_start,pred_sub_end=self.subject_tagger(input_ids)
        #batch_size * max_seq_len * 1

        chosen_sub_start,chosen_sub_end=chosen_sub
        
        chosen_sub_start_feature=torch.gather(score,dim=1,index=chosen_sub_start)
        chosen_sub_end_feature=torch.gather(score,dim=1,index=chosen_sub_end)
        #batch_size * max_seq_len * word_embed_size

        chosen_sub_feature=(chosen_sub_start_feature+chosen_sub_end_feature)/2

        obj_tagger_input_ids=input_ids+chosen_sub_feature
        #batch_size * max_seq_len * word_embed_size
        
        pred_obj_start,pred_obj_end=self.object_tagger(obj_tagger_input_ids)
        #batch_size * max_Seq_len * relation_nums

        #除去Padding部分,返回输出结果
        pred_sub_start,pred_sub_end=pred_sub_start.unsqueeze(2),pred_sub_end.unsquezze(2)
        #[batch_size * max_Seq_len]
        mask=attention_mask==1
        pred_sub_start=pred_sub_start.masked_select(mask)
        pred_sub_end=pred_sub_end.masked_select(mask)
        # list

        mask=mask.squeeze(2).expand(-1,-1,self.relation_nums)
        # [batch_size * max_seq_len * relation_nums]
        pred_obj_start=pred_obj_start.masked_select(mask)
        pred_obj_end=pred_obj_end.masked_select(mask)
        # list [0~1]

        return pred_sub_start,pred_sub_end,pred_obj_start,pred_obj_end
    
    def cal_loss(self,pred_sub_vec,sub_vec,pred_obj_vec,obj_vec):

        pred_sub_start,pred_sub_end=pred_sub_vec
        target_sub_start,target_sub_end=sub_vec

        pred_obj_start,pred_obj_end=pred_obj_vec
        target_obj_start,target_obj_end=obj_vec
        

        sub_start_loss=self.loss_layer(pred_sub_start,target_sub_start)
        sub_end_loss=self.loss_layer(pred_sub_end,target_sub_end)
        obj_start_loss=self.loss_layer(pred_obj_start,target_obj_start)
        obj_end_loss=self.loss_layer(pred_obj_end,target_obj_end)

        return (sub_start_loss+sub_end_loss) + (obj_start_loss,obj_end_loss)
    
    def prdict(self,input_ids,attention_mask,chosen_sub,threshold):
        pred_sub_start,pred_sub_end,pred_obj_start,pred_obj_end=self.forward(input_ids=input_ids,\
            attention_mask=attention_mask,chosen_sub=chosen_sub)
        
        one=torch.from_numpy(np.array(1))
        zero=torch.from_numpy(np.array(0))
        #[batch_size,max_seq_len]
        F=lambda x : torch.where( x >threshold,one,zero)
        pred_sub_start=F(pred_sub_start)
        pred_sub_end=F(pred_sub_end)
        pred_obj_start=F(pred_obj_start)
        pred_obj_end=F(pred_obj_end)

        return pred_sub_start,pred_sub_end,pred_obj_start,pred_obj_end
        #list 0 | 1
        

class Tagger(nn.Module):
    def __init__(self,word_emb_size,activate_function,out_size):
        super(Tagger,self).__init__()
        self.tagger_start=nn.Linear(word_emb_size,out_size)
        self.tagger_end=nn.Linear(word_emb_size,out_size)
        self.activate=activate_function
    def forward(self,input_ids):
        '''
        input_ids: [batch_size * max_len_seq * word_emb_size]
        '''
        score_start=self.activate(self.tagger_start(input_ids))
        score_end=self.activate(self.tagger_end(input_ids))
        return score_start,score_end

    def cal_loss(self,input_ids):

        return input_ids



