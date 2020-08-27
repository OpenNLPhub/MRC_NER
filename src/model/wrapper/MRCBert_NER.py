import torch
import numpy as np
import torch.optim as optim
from src.model.wrapper.BaseWrapper import BaseWrapper
from src.model.nn.MRCBert import MRCBert
from src.config.ModelConfig import BertConfig
from src.utils.common import overrides,flatten_lists
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
        self.optimizer=optim.Adam(self.model.parameters(),lr=self.lr)
    
    @overrides(BaseWrapper)
    def _cal_loss(self,batch_data,**kwargs):
        tokenizer=kwargs.get('tokenizer',None)
        if tokenizer==None:
            raise ValueError('Need Tokenizer!')
        
        input_ids,attention_mask,token_type_ids,tags_lists=batch_data
        # transform data to tensor
        input_ids=torch.from_numpy(np.array(input_ids)).long().to(self.device)
        attention_mask=torch.from_numpy(np.array(attention_mask)).long().to(self.device)
        token_type_ids=torch.from_numpy(np.array(token_type_ids)).long().to(self.device)
        tags=torch.from_numpy(np.array(flatten_lists(tags_lists))).to(self.device)
        score=self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,\
            berttokenizer=tokenizer)
        # import pdb; pdb.set_trace()
        loss=self.model.cal_loss(score,tags)
        return loss

    @overrides(BaseWrapper)
    def test(self,test_data_loader,**kwargs):
        pass
