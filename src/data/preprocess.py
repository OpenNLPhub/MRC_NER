'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-08-24 23:05:58
 * @desc 
'''

import os
import sys

import random
from collections import Counter
from tqdm import tqdm
import operator
import src.config.args as args
from src.utils.common import overrides
import json
def line2char(line):
    '''
    :param line: 原始行 字:label
    :return: (单词，标签)
    '''
    res=line.strip('\n').split()
    return res

def train_dev_split(x, y, shuffle=True, valid_size=0.2 ,random_state=2020):
    '''
    :param X: sent_list
    :param y:labels
    :param random_state: 随机种子
    '''
    data=[]
    print('Split the training data and dev data')
    for data_x,data_y in tqdm(zip(x,y)):
        data.append((data_x,data_y))
    
    N=len(data)
    test_size=int(N*valid_size)

    if shuffle:
        random.seed(random_state)
        random.shuffle(data)
    
    dev=data[:test_size]
    train=data[test_size:]

    return dev,train

def build_vocab(min_freq=1,stop_word_list=None):
    """
    建立词典
    :param min_freq: 最小词频限制
    :param stop_list: 停用词 @type：file_path
    :return: vocab
    """
    count=Counter()

    with open(os.path.join(args.RAW_SOURCE_DATA,'train_data'),'r') as f:
        print('Building vocab')
        for line in tqdm(f.readlines()):
            t=line2char(line)
            if len(t)==0:continue
            word,label=t
            count.update(word)
    if stop_word_list:
        stop_list={}
        with open(args.STOP_WORD_LIST,'r') as f:
            for i,line in enumerate(f):
                word=line.strip('\n')
                if stop_list.get(word) is None:
                    stop_list[word]=i
        count={k:v for k,v in count.items() if k not in stop_list}

    count=sorted(count.items(),key=operator.itemgetter(1))
    vocab=[w[0] for w in count if w[1]>=min_freq]

    vocab=args.FLAG_WORDS+vocab

    assert vocab[0]=='[PAD]'

    with open(args.VOCAB_FILE,'w') as f:
        for w in vocab:
            f.write(w+'\n')
        


def produce_data_flat_ner():
    '''
    train,dev集需要自己划分
    '''
    print('process raw training data')
    with open(os.path.join(args.RAW_SOURCE_DATA,'train_data'),'r') as f:
        sent_list,tags_list=[],[]
        sent,tags=[],[]
        for line in tqdm(f.readlines()):
            t=line2char(line)
            if len(t)==0:
                sent_list.append(sent)
                tags_list.append(tags)
                sent=[]
                tags=[]
            else:
                # import pdb
                # pdb.set_trace()
                char,tag=t
                sent.append(char)
                tags.append(tag)
    dev,train=train_dev_split(sent_list,tags_list)
    # import pdb
    # pdb.set_trace()
    '''
    写入文件格式
    李 华 爱 北 京\t B-PER I-PER O B-LOC I-LOC\n
    '''
    with open(os.path.join(args.FLAT_SOURCE_DATA,'dev_data.txt'),'w') as dev_txt,\
        open(os.path.join(args.FLAT_SOURCE_DATA,'train_data.txt'),'w') as train_txt:
        print('write train data')
        for sent,tags in tqdm(train):
            ans=' '.join(sent)+'\t'+' '.join(tags)+'\n'
            train_txt.write(ans)
        print('write dev data')
        for sent,tags in tqdm(dev):
            ans=' '.join(sent)+'\t'+' '.join(tags)+'\n'
            dev_txt.write(ans)
            

    with open(os.path.join(args.RAW_SOURCE_DATA,'test_data'),'r') as f:
        sent_list,tags_list=[],[]
        sent,tags=[],[]
        for line in tqdm(f.readlines()):
            t=line2char(line)
            if len(t)==0:
                sent_list.append(sent)
                tags_list.append(tags)
                sent=[]
                tags=[]
            else:
                char,tag=t
                sent.append(char)
                tags.append(tag)
    
    with open(os.path.join(args.FLAT_SOURCE_DATA,'test_data.txt'),'w') as test_txt:
        print('write test data')
        for sent,tags in tqdm(zip(sent_list,tags_list)):
            ans=' '.join(sent)+'\t'+' '.join(tags)+'\n'
            test_txt.write(ans) 


def _write_mrc_data(label,data,file,prefix):
    assert prefix in ['train','dev','test']
    print('Label:{} \t {} data'.format(label,prefix))

    for sent,tags in tqdm(data):
        sent_str=' '.join(sent)
        tran_tags=['O' if tag.find(label)==-1 else tag.split('-')[0] for tag in tags]
        #丢弃无实体的data item
        if len(set(tran_tags))==1 and tran_tags[0]=='O': continue
        tags_str=' '.join(tran_tags)
        ans_str=sent_str+'\t'+tags_str+'\n'
        file.write(ans_str)
    

'''
针对MRC nested NER 嵌套问题的数据集。
如果我们用MRC，就需要将每一个Label做一个数据集，在label中的标签 只有 ['B','I','O']
'''
def produce_data_mrc():
    '''
    train,dev集需要自己划分
    '''
    print('process raw training data')
    with open(os.path.join(args.RAW_SOURCE_DATA,'train_data'),'r') as f:
        sent_list,tags_list=[],[]
        sent,tags=[],[]
        for line in tqdm(f.readlines()):
            t=line2char(line)
            if len(t)==0:
                assert len(sent) ==len(tags)
                sent_list.append(sent)
                tags_list.append(tags)
                sent=[]
                tags=[]
            else:
                char,tag=t
                # 这里做了一个简单的映射，如果空格作为一个打了tag的词要输入，我们将其映射为 -
                # 因为在BertTokenizer 中 空格被认为是一个分隔符，不会对空格打标签，会造成，tag和data 不对齐
                if char ==' ':
                    char='-'
                sent.append(char)
                tags.append(tag)
    dev,train=train_dev_split(sent_list,tags_list)


    with open(os.path.join(args.RAW_SOURCE_DATA,'test_data'),'r') as f:
        sent_list,tags_list=[],[]
        sent,tags=[],[]
        for line in tqdm(f.readlines()):
            t=line2char(line)
            if len(t)==0:
                assert len(sent) ==len(tags)
                sent_list.append(sent)
                tags_list.append(tags)
                sent=[]
                tags=[]
            else:
                char,tag=t
                if char ==' ':
                    char='-'
                sent.append(char)
                tags.append(tag)
            
    test=list(zip(sent_list,tags_list))

    labels=args.MRC_LABEL
    labels.remove('O')
    # import pdb
    # pdb.set_trace()
    for label in labels:
        with open(os.path.join(args.MRC_SOURCE_DATA,label+'_train.txt'),'w') as train_txt,\
            open(os.path.join(args.MRC_SOURCE_DATA,label+'_dev.txt'),'w') as dev_txt,\
            open(os.path.join(args.MRC_SOURCE_DATA,label+'_test.txt'),'w') as test_txt:

            _write_mrc_data(label,train,train_txt,'train')
            _write_mrc_data(label,dev,dev_txt,'dev')
            _write_mrc_data(label,test,test_txt,'test')

# def produce_data_mrc():

class InputUnit(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """创建一个输入实例
        Args:
            guid: 每个example拥有唯一的id
            text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a lists [str_1,str2]
            text_b: 第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
            label: example对应的标签，对于训练集和验证集应非None，测试集为None
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    '''
    数据预处理基类，可自定义集成该类
    '''
    def get_train_units(self,data_dir):
        '''
        读取训练集
        '''
        raise NotImplementedError()
    
    def get_dev_units(self,data_dir):
        '''
        读取验证集
        '''
        raise NotImplementedError()

    def get_test_units(self,data_dir):
        '''
        读取测试集
        '''
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()
    
    @classmethod
    def _read_line(cls,input_file):
        with open(input_file,"r",encoding='utf-8') as f:
            sent_list,tags_list=[],[]
            for line in f.readlines():
                sent,tags=line.strip('\n').split('\t')
                sent_list.append(sent.split(' '))
                tags_list.append(tags.split(' '))
                assert len(sent_list)==len(tags_list)
            return (sent_list,tags_list)
    

class FlatDataProcessor(DataProcessor):
    '''
    构成Inputunit 模式
    FlatDataProcessor 可以读取数据
    - data_dir
    |- train.txt
    |- dev.txt
    |- 
    其中train.txt dev.txt 皆为
    李 华 在 北 京\t B-PER I-PER O B-LOC I-LOC
    '''

    train='train.txt'
    dev='dev.txt'
    test='test.txt'
    def __init__(self):
        super(FlatDataProcessor,self).__init__()
    
    @overrides(DataProcessor)
    def get_train_units(self,data_dir):
        units=[]
        path=os.path.join(data_dir,FlatDataProcessor.train)
        pairs=self._read_line(path)
        return units.append(self._create_unit(pairs,"train"))
    
    @overrides(DataProcessor)
    def get_dev_units(self,data_dir):
        units=[]
        path=os.path.join(data_dir,FlatDataProcessor.dev)
        pairs=self._read_line(path)
        return units.append(self._create_unit(pairs,"dev"))
    
    @overrides(DataProcessor)
    def get_test_units(self, data_dir):
        units=[]
        path=os.path.join(data_dir,FlatDataProcessor.test)
        pairs=self._read_line(path)
        return units.append(self._create_unit(pairs,"test"))

    @overrides(DataProcessor)
    def get_labels(self,data_dir):
        return args.LABELS
    
    def _create_unit(self,pairs,set_type,label_text=None,label=None):
        #与MRC接口相同
        #sent_list=['李','华','在','北','京']
        sent_list,tags_list=pairs
        sent_list=[' '.join(i) for i in sent_list]
        # 这里用 空格 进行 分词预处理
        t= self.__class__.__name__ if label==None else label
        guid="{}-{}".format(set_type,t)
        unit=InputUnit(guid=guid,text_a=sent_list,label=tags_list)
        return unit


class MRCDataProcessor(DataProcessor):
    '''
    构成Inputunit 模式
    FlatDataProcessor 可以读取数据
    - data_dir
    |- {label}_train.txt
    |- {label}_dev.txt
    |- description.json

    其中PER_train.txt PER_dev.txt 皆为
    李 华 在 北 京\t B I O O O
    
    其中description.json 
    {
        'PER':'找到人名',
        'LOC':'找到地名'
    }

    '''
    MRC_desc='description.json'
    MRC_train='_train.txt'
    MRC_dev='_dev.txt'
    MRC_test='_test.txt'
    def __init__(self):
        super(MRCDataProcessor,self).__init__()

    @overrides(DataProcessor)
    def get_train_units(self,data_dir):
        units=[]
        with open(os.path.join(data_dir,MRCDataProcessor.MRC_desc),'r',encoding='utf-8') as f:
            label2desc=json.loads(f.read())
        labels=label2desc.keys()
        for label in labels:
            path=os.path.join(data_dir,label+MRCDataProcessor.MRC_train)
            pairs=self._read_line(path)
            label_text=label2desc[label]
            units.append(self._create_unit(pairs,'train',label_text))
        return units
    
    @overrides(DataProcessor)
    def get_dev_units(self,data_dir):
        units=[]
        with open(os.path.join(data_dir,MRCDataProcessor.MRC_desc),'r',encoding='utf-8') as f:
            label2desc=json.loads(f.read())
        labels=label2desc.keys()
        for label in labels:
            path=os.path.join(data_dir,label+MRCDataProcessor.MRC_dev)
            pairs=self._read_line(path)
            label_text=label2desc[label]
            units.append(self._create_unit(pairs,'dev',label_text))
        return units
    
    @overrides(DataProcessor)
    def get_test_units(self, data_dir):
        units=[]
        with open(os.path.join(data_dir,MRCDataProcessor.MRC_desc),'r',encoding='utf-8') as f:
            label2desc=json.loads(f.read())
        labels=label2desc.keys()
        for label in labels:
            path=os.path.join(data_dir,label+MRCDataProcessor.MRC_test)
            pairs=self._read_line(path)
            label_text=label2desc[label]
            units.append(self._create_unit(pairs,'test',label_text))
        return units
    
    
    @overrides(DataProcessor)
    def get_labels(self,data_dir):
        with open(os.path.join(data_dir),MRCDataProcessor.MRC_desc) as f:
            label2desc=json.loads(f.read())
        return list(label2desc.keys())

    def _create_unit(self,pairs,set_type,label_text=None,label=None):
        #label_path label description txt
        #sent_list=['李','华','在','北','京']
        sent_list,tags_list=pairs
        sent_list=[' '.join(i) for i in sent_list]
        # 这里用 空格 进行 分词预处理
        t= self.__class__.__name__ if label==None else label
        guid="{}-{}".format(set_type,t)
        text_b=[label_text for i in range(len(sent_list))]
        unit=InputUnit(guid=guid,text_b=text_b,label=tags_list,text_a=sent_list)
        return unit
        


def convert_units_to_features(units,label_list,tokenizer):
    '''
    这里tokenizer 全部使用 BertTokenizer
    '''
    label_map={label:i for i,label in enumerate(label_list)}
    text_a=[]
    text_b=[]
    tags_list=[]
    # import pdb;pdb.set_trace()
    for unit in units:
        text_a.extend(unit.text_a)
        text_b.extend(unit.text_b)
        tags_list.extend(unit.label)
    assert len(text_a)==len(text_b)
    
    #这里 我们不能用tokenizer 直接生成 input ，在bert tokenizer中 “中国IBM” 会被分为 “中”，“国“，”IBM"
    #但是在数据集中应该是 “中”，“国“，”I“，“B”，”M"。这回造成tag和label不匹配
    inputs=tokenizer(text_a,text_b,padding=True)
    #inputs ={input_ids,attention_mask,token_type_ids}
    
    '''
    Debug

    for i,value in enumerate(text_a):
        inputs=tokenizer(value)
        input_ids=inputs['input_ids']
        if len(input_ids)-2!=len(tags_list[i]):
            print(value)
            print(input_ids)
            print(tags_list[i])
            import pdb;pdb.set_trace()
    '''
    
    # len_list=[ for i in enumerate(zip)]
    tag_id_list=[]
    # batch_size * seq_len     其中seq_len 不是一个定值
    for i,l in enumerate(tags_list):
        t=[]
        for j,tag in enumerate(l):
            t.append(label_map.get(tag))
        tag_id_list.append(t)
    # import pdb;pdb.set_trace()
    #tag_id_list 逻辑上应该flat之后在进行 loss计算，这里因为考虑到分批的操作，没有将他变成tensor
    
    return inputs,tag_id_list


if __name__=='__main__':
    # build_vocab()
    # produce_data_flat_ner()
    produce_data_mrc()               