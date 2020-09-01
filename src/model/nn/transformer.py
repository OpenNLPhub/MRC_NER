'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-08-31 11:23:29
 * @desc 
    The model that descriped in Attention is all you need
    Coding According to 
    https://state-of-art.top/archives/
    The Annotated Transformer from Harvard
    http://nlp.seas.harvard.edu/2018/04/03/attention.html#model-architecture
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math

def make_model(src_vocab,tag_vocab,N=6,d_model=512,d_ff=2058,h=8,dropout=0.1):
    '''
    Implementation to get Transformer
    Params:
        src_vocab source data vocabulary list's size
        tag_vocab : target data vocabulary list's size
        N: encoderlayer in Encoder
        d_model: word embedding size
        d_ff: full connected layer size
        h: the num of multiheadedAttention Layer's head
        dropout: dropout probability
    '''
    c=copy.deepcopy
    # it is important to deepcopy the module and diliver it into the function
    attn= MultiHeadedAttention(h,d_model,dropout=dropout)
    ff=PositionwiseFeedForward(d_model,d_ff,dropout=dropout)
    position=PositionalEncoding(d_model,dropout=dropout)
    encoder=Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),N=N)
    decoder=Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout=dropout),N=N)
    src_embed=nn.Sequential(Embeddings(d_model,src_vocab),c(position))
    tgt_embed=nn.Sequential(Embeddings(d_model,tag_vocab),c(position))
    model=EncoderDecoder(encoder,decoder,src_embed,tgt_embed,Generator(d_model,tag_vocab))

    #Initialize parameters with Glorot
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform(p)
            
    return model

'''
Encoder-Decoder Abstract Model
'''

class EncoderDecoder(nn.Module):
    '''
    A standard Encoder-Decoder architecture. Base for this and many other models.
    '''
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        '''
        Params:
        encoder:Encoder Model
        decoder:Decoer Model
        src_mebed:training word embedding model
        tgt_embed:target word embedding model. Difference between src and tgt is the mask
                In tgt_embed we should predict the word one by one in sequence.
        generator: the final classifer for the model to choose word.
        '''
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.generator=generator

    def forward(self,src,tgt,src_mask,tgt_mask):
        '''
        Take in and process masked src and target sequences.
        '''
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)
    
    def encode(self,src,src_mask):
        '''
        Consider padding in batch, we have to put in src_mask
        '''
        return self.encoder(self.src_embed(src),src_mask)
    
    def decode(self,memory,src_mask,tgt,tgt_mask):
        '''
        Params:
            memory: the output of encoder
            src_mask: padding mask in encoder
        '''
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step"

    def __init__(self,d_model,vocab):
        '''
        params:
            d_model: the dimension of  hidden layer
            vocab: the size of vocabulary list
        '''
        super(Generator,self).__init__()
        self.proj=nn.Linear(d_model,vocab)
    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)


'''
Encoder:

The encoder is composed of a stack of n=6 identical layers
'''

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self,layer,N):
        '''
        Abstract Encoder in which layer architecture and layer num can be changed
        '''
        super(Encoder,self).__init__()
        self.layers=clones(layer,N) #list of layers
        self.norm=LayerNorm(layer.size)
    
    def forward(self,x,mask):
        "Pass the input (and Mask through each layer in turn)."
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)



class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        '''
        Layer Normalization
        '''
        super(LayerNorm,self).__init__()
        self.a_2=nn.Parameter(torch.ones(features))
        self.b_2=nn.Parameter(torch.zeros(features))
        self.eps=eps
    def forward(self,x):
        mean=x.mead(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2

'''
To facilitate these residual connections all sub-layers in the model,
as well as the mebedding layers,prodcuce outputs of dimension d_model=512
'''

class SublayerConnection(nn.Module):
    '''
    A residual connection follwed by a layer norm
    Add & Norm layer
    '''
    def __init__(self,size,dropout):
        super(SublayerConnection,self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(p=dropout)
    
    def forward(self,x,sublayer):
        '''
        Apply residual connection to any sublayer with the same input size
        '''
        return x+self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(EncoderLayer,self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SublayerConnection(size,dropout),2)
        self.size=size
    
    def forward(self,x,mask):
        #go through self_attention layer (key query value)  and a Add & Norm Layer
        #There we use lambda to construct a f(x) in which mask is a constant
        # we deliver this function into SublayerConnection forward()
        x=self.sublayer[0](x,lambda x: self.self_attn(x,x,x,mask))
        x=self.sublayer[1](x,self.feed_forward)
        return x



'''
Decoder 
which is also composed of a stack of N=6 identical layers
'''

class Decoder(nn.Module):
    def __init__(self,layer,N):
        '''
        Abstract Decoder
        '''
        super(Decoder,self).__init__()
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)

    def forward(self,x,memory,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super(DecoderLayer,self).__init__()
        self.size=size
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SublayerConnection(size,dropout),3)
    
    def forward(self,x,memory,src_mask,tgt_mask):
        m=memory
        #firstly  go through a self-attention SublayerConnection
        x=self.sublayer[0](x,lambda x: self.self_attn(x,x,x,tgt_mask))

        #secondly go through a attention SublayerConnection 
        #Query is from tgt, but Key and Value is from memory of encoder
        x=self.sublayer[1](x,lambda x:self.self_attn(x,m,m,src_mask))

        #finally go through a feed-forward layer
        x=self.sublayer[2](x,self.feed_forward)

        return x

'''
Create Position Mask
This mask conbined with fact that the output embeddings are offset by one position,
ensures that the predictions for position i can be depend only on the known outputs at positions less than i
'''
def subsequent_mask(size):
    '''Mask out subsequent position'''
    attn_shape=(1,size,size)
    subsequent_mask=np.triu(np.ones(attn_shape),k=1).astype('uint8')
    '''
    if size=4
     [[
        [0,1,1,1],
        [0,0,1,1],
        [0,0,0,1],
        [0,0,0,0],
     ]]
    '''
    return torch.from_numpy(subsequent_mask)==0

    '''
    [[
        [T,F,F,F],
        [T,T,F,F],
        [T,T,T,F],
        [T,T,T,T]
    ]]
    '''

'''
------------------------Attention Mechanism---------------------------
'''

'''
Calculate Q-K-V
Exmaple 
Q (batch_size,heads,max_seq_len,d_k)
K (batch_size,heads,max_seq_len,d_k)
V (batch_size,heads,max_seq_len,d_v)
d_v=d_k
'''
def attention(query,key,value,mask=None,dropout=None):
    d_k=query.size(-1)
    score=torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)
    # divide by sqrt(d_k) in order to keep gradient stable
    '''
    (batch_size,head,max_seq_len, d_k) * (batch_size,head,max_seq_len,d_k)
    (batch_size,head,max_seq_len,max_seq_len) 
    '''
    
    if mask is not None:
        #For padding word, the valid is 0
        scores=scores.masked_fill(mask==0,-1e9)
    
    #use softmax to transfer weight into [0,1]
    #(batch_size,head,max_seq_len,max_seq_len)
    p_attn=F.softmax(scores,dim=-1)

    if dropout is not None:
        p_attn=dropout(p_attn)
    
    #(batch_size,head,max_seq_len,max_seq_len) (batch_size,head,max_seq_len,d_v)
    # (batch_size,head,max_seq_len,d_v)
    score=torch.matmul(p_attn,value)
    
    return score,p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        super(MultiHeadedAttention,self).__init__()
        assert d_model % h==0
        #In reality, MultiHeaded Attention have h*3 Linear，
        #But in this code, the author decrease the size of d_k,

        self.d_k=d_model // h
        self.h=h
        self.linears=clones(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)
    
    def forward(self,query,key,value,mask=None):
        # mask (batch_size,max_len_seq)
        if mask is not None:
            mask=mask.unsqueeze(1)
        # mask (batcj_size,1, max_len_seq)
        nbatches=query.size(0)

        '''
        self.linears[0](query)-> (batch_size,max_len_seq,n_model)->
        (batch_size,max_len_seq,h,d_k) -> (batch_size,h,max_len_seq,d_k)
        '''
        query=self.linears[0](query).view(nbatches,-1,self.h,self.d_k).transpose(1,2)

        key=self.linears[1](key).view(nbatches,-1,self.h,self.d_k).transpose(1,2)

        value=self.linears[2](value).view(nbatches,-1,self.h,self.d_k).transpose(1,2)

        #calculate attention

        # batch_size,h,max_len_seq,d_k
        x,self.attn=attention(query,key,value,mask=mask,dropout=self.dropout)


        # concat this into n_model and apply a final linear
        '''
        (batch_size,h,max_len_seq,d_k) -> (batch_size,max_len_seq,h,d_k) ->
        (batch_size,max_len_seq,d_model)
        '''
        x=x.transpose(1,2).contiguous().view(nbatches,-1,self.d_h*self.h)

        return self.linears[-1](x)


'''
Feed-Forward Networks
'''

class PositionwiseFeedForward(nn.Module):

    def __init__(self,d_model,d_ff,dropout=0.1):
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

'''
Embedding
'''

class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        super(Embeddings,self).__init__()
        self.lut=nn.Embedding(vocab,d_model)
        self.d_model=d_model
    
    def forward(self,x):
        return self.lut(x)*math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
   
    def forward(self, x):
        p=self.pe[:,:x.size(1)].requires_grad=False
        x = x + p
        return self.dropout(x)


    