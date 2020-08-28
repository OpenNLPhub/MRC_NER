from torch import nn
import torch
import numpy as np
import torch.optim as optim
from transformers import BertModel,BertConfig
import src.config.ModelConfig as ModelConfig
import torch.nn.functional as F
from src.utils.common import flatten_lists
class MRCBert(nn.Module):
    def __init__(self,num_labels,vocab_size=0,emb_size=0,hidden_size=0,use_pretrained=True):
        super(MRCBert,self).__init__()
        if use_pretrained:
            self.bertconfig=BertConfig.from_pretrained(ModelConfig.BERT_BASE_CHINESE,author='lingze',num_labels=num_labels)
            self.bert=BertModel.from_pretrained(ModelConfig.BERT_BASE_CHINESE,config=self.bertconfig)
            self.vocab_size=self.bertconfig.vocab_size
            self.emb_size=self.bertconfig.hidden_size
            self.hidden_size=self.bertconfig.hidden_size
        else:
            self.vocab_size=vocab_size
            self.emb_size=emb_size
            self.hidden_size=hidden_size
            self.bertconfig=BertConfig(vocab_size=self.vocab_size,hidden_size=self.hidden_size,\
                num_labels=num_labels,author='lingze')
            self.bert=BertModel(config=self.bertconfig)
        self.dropout=nn.Dropout(self.bertconfig.hidden_dropout_prob)
        self.ll=nn.Linear(self.hidden_size,num_labels)
    
    def forward(self,input_ids,attention_mask,token_type_ids,berttokenizer):
        '''
        input_ids:输入数据 [Batch_size,max_length]
        token_type_ids: 标定word属于哪个句子，只能取1,0.
        attention_mask:Mask to avoid performing attention on padding token indices. Mask Value selected in [0,1] [Batch_size,max_length]
        '''
        SEP=berttokenizer.convert_tokens_to_ids('[SEP]')
        CLS=berttokenizer.convert_tokens_to_ids('[CLS]')

        hidden_score=self.bert(input_ids=input_ids,\
            attention_mask=attention_mask,token_type_ids=token_type_ids)[0]
        #[batch_size,sent_len,hidden_size]
        # import pdb; pdb.set_trace()
        #这里我们需要利用token_type_ids 丢掉Query Representation,同时可以去掉Padding部分
        mask=(token_type_ids==0) & (attention_mask==1) & (input_ids !=SEP) & (input_ids !=CLS)

        mask=mask.unsqueeze(2).expand(-1,-1,self.hidden_size)
        hidden_score=hidden_score.masked_select(mask).reshape(-1,self.hidden_size)
        hidden_score=self.dropout(hidden_score)
        #加入Linear层
        return self.ll(hidden_score)

    
    #labels [batch_size,seq_len] 不加padding
    def cal_loss(self,score,labels,weight):
        flatten_labels=flatten_lists(labels)
        return F.cross_entropy(score,labels,weight=weight)
    

    def predict(self,input_ids,attention_mask,token_type_ids,berttokenizer):
        
        score=self.forward(input_ids=input_ids,attention_mask=attention_mask,\
            token_type_ids=token_type_ids,berttokenizer=berttokenizer)
        #score [flatten_size, num_labels] 可以通过attention_mask,token_type_ids将其还原
        labels=torch.argmax(score,dim=1)
        #labels [flatten_size]
        return labels





    
