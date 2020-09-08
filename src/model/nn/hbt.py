
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class HBT(nn.Module):
    def __init__(self,relation_num,bertmodel):
        super(HBT,self).__init__()
        self.bert=bertmodel
        self.word_emb_size=bertmodel.config.hidden_size
        self.subject_tagger=Tagger(self.word_emb_size,torch.sigmoid,1)
        self.object_tagger=Tagger(self.word_emb_size,torch.sigmoid,relation_num)
        self.loss_layer=nn.BCELoss()
        self.relation_num=relation_num
        
        self.one=nn.Parameter(torch.ones(1))
        self.one.requires_grad=False
        self.zero=nn.Parameter(torch.zeros(1))
        self.zero.requires_grad=False
        self.weight=nn.Parameter(torch.from_numpy(np.array([0.5,1.5])))
        self.weight.requires_grad=False

        self.layernorm=nn.LayerNorm(self.word_emb_size,eps=1e-12)
        self.dropout=nn.Dropout(p=0.1)

    def forward(self,input_ids,attention_mask,chosen_sub):
        score,cls_score=self.bert(input_ids=input_ids,attention_mask=attention_mask)
        #batch_size * max_seq_len * word_emb_size
        #batch_size * word_emb_size
        # import pdb
        # pdb.set_trace()
        pred_sub_start,pred_sub_end=self.subject_tagger(score)
        #batch_size * max_seq_len * 1

        chosen_sub_start,chosen_sub_end=chosen_sub
        
        chosen_sub_start_feature=torch.gather(score,dim=1,index=chosen_sub_start)
        chosen_sub_end_feature=torch.gather(score,dim=1,index=chosen_sub_end)
        #batch_size * max_seq_len * word_embed_size

        chosen_sub_feature=(chosen_sub_start_feature+chosen_sub_end_feature)/2

        obj_tagger_input_ids=score+chosen_sub_feature
        #batch_size * max_seq_len * word_embed_size
        obj_tagger_input_ids=self.layernorm(obj_tagger_input_ids)

        pred_obj_start,pred_obj_end=self.object_tagger(obj_tagger_input_ids)
        #batch_size * max_Seq_len * relation_nums


        #除去Padding部分,返回输出结果
        pred_sub_start,pred_sub_end=pred_sub_start.squeeze(-1),pred_sub_end.squeeze(-1)
        #[batch_size * max_Seq_len]
        mask=attention_mask==1
        pred_sub_start=pred_sub_start.masked_select(mask)
        pred_sub_end=pred_sub_end.masked_select(mask)
        # list

        mask=mask.unsqueeze(2).expand(-1,-1,self.relation_num)
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
        
        #import pdb;pdb.set_trace()
        F=lambda x,y,weights: weighted_binary_cross_entropy(x,y,weights=weights)
        sub_start_loss=F(pred_sub_start,target_sub_start,None)
        sub_end_loss=F(pred_sub_end,target_sub_end,None)
        
        obj_start_loss=F(pred_obj_start,target_obj_start,self.weight)
        obj_end_loss=F(pred_obj_end,target_obj_end,self.weight)

        print("sub_start:{}\t sub_end:{}\t obj_start:{}\t obj_end:{}"\
            .format(sub_start_loss.item(),sub_end_loss.item(),obj_start_loss.item(),obj_end_loss.item()))
        
        return sub_start_loss+sub_end_loss+obj_start_loss+obj_end_loss
        # return obj_start_loss+obj_end_loss
    
    def predict(self,input_ids,attention_mask,chosen_sub,threshold):
        pred_sub_start,pred_sub_end,pred_obj_start,pred_obj_end=self.forward(input_ids=input_ids,\
            attention_mask=attention_mask,chosen_sub=chosen_sub)
        
        one=self.one
        zero=self.zero
        #[batch_size,max_seq_len]
        
        F=lambda x : torch.where(x>threshold,one,zero).long()
        pred_sub_start=F(pred_sub_start)
        pred_sub_end=F(pred_sub_end)
        pred_obj_start=F(pred_obj_start)
        pred_obj_end=F(pred_obj_end)

        return pred_sub_start,pred_sub_end,pred_obj_start,pred_obj_end
        #list 0 | 1


def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
            weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


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
        # import pdb
        # pdb.set_trace()
        score_start=self.activate(self.tagger_start(input_ids))
        score_end=self.activate(self.tagger_end(input_ids))
        return score_start,score_end



