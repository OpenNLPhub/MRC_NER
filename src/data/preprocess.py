import os
import sys
import random
from collections import Counter
from tqdm import tqdm
import operator
import config.args as args
from utils.common import overrides
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


def produce_data_flat_ner():
    '''
    train,dev集需要自己划分
    '''
    print('process raw training data')
    with open(os.path.join(args.RAW_SOURCE_DATA,'train_data')) as f:
        sent_list,tags_list=[],[]
        sent,tags=[],[]
        for line in tqdm(f.readlines()):
            t=line2char(line)
            if len(t)!=0:
                sent_list.append(sent)
                tags_list.append(tags)
            else:
                char,tag=t
                sent.append(char)
                tags.append(tag)
    dev,train=train_dev_split(sent_list,tags_list)
    
    '''
    写入文件格式
    李 华 爱 北 京\t B-PER I-PER O B-LOC I-LOC\n
    '''
    with open(os.path.join(args.FLAT_SOURCE_DATA,'dev_data.txt')) as dev_txt,\
        open(os.path.join(args.FLAT_SOURCE_DATA,'train_data.txt')) as train_txt:
        print('write train data')
        for sent,tags in tqdm(train):
            ans=' '.join(sent)+'\t'+' '.join(tags)+'\n'
            train_txt.write(ans)
        print('write dev data')
        for sent,tags in tqdm(dev):
            ans=' '.join(sent)+'\t'+' '.join(tags)+'\n'
            dev_txt.write(ans)
            

    with open(os.path.join(args.RAW_SOURCE_DATA,'test_data')) as f:
        sent_list,tags_list=[],[]
        sent,tags=[],[]
        for line in tqdm(f.readlines()):
            t=line2char(line)
            if len(t)!=0:
                sent_list.append(sent)
                tags_list.append(tags)
            else:
                char,tag=t
                sent.append(char)
                tags.append(tag)
    
    with open(os.path.join(args.FLAT_SOURCE_DATA,'test_data.txt')) as test_txt:
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
    with open(os.path.join(args.RAW_SOURCE_DATA,'train_data')) as f:
        sent_list,tags_list=[],[]
        sent,tags=[],[]
        for line in tqdm(f.readlines()):
            t=line2char(line)
            if len(t)!=0:
                sent_list.append(sent)
                tags_list.append(tags)
            else:
                char,tag=t
                sent.append(char)
                tags.append(tag)
    dev,train=train_dev_split(sent_list,tags_list)

    with open(os.path.join(args.RAW_SOURCE_DATA,'test_data')) as f:
        sent_list,tags_list=[],[]
        sent,tags=[],[]
        for line in tqdm(f.readlines()):
            t=line2char(line)
            if len(t)!=0:
                sent_list.append(sent)
                tags_list.append(tags)
            else:
                char,tag=t
                sent.append(char)
                tags.append(tag)
    test=zip(sent_list,tags_list)

    labels=args.MRC_LABEL
    labels.remove('O')
    
    for label in labels:
        with open(os.path.join(args.MRC_SOURCE_DATA,label+'_train.txt')) as train_txt,\
            open(os.path.join(args.MRC_SOURCE_DATA,label+'_dev.txt')) as dev_txt,\
            open(os.path.join(args.MRC_SOURCE_DATA,label+'_test.txt')) as test_txt:

            _write_mrc_data(label,train,train_txt,'train')
            _write_mrc_data(label,dev,dev_txt,'dev')
            _write_mrc_data(label,test,test_txt,'test')

# def produce_data_mrc():

class InputUnit(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """创建一个输入实例
        Args:
            guid: 每个example拥有唯一的id
            text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a
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
            return zip(sent_list,tags_list)



class FlatDataProcessor(DataProcessor):
    '''
    构成Inputunit 模式
    '''
    def __init__(self):
        super(FlatDataProcessor,self).__init__()
    
    @overrides
    def get_train_units(self,data_dir):
        path=os.path.join(data_dir,'train.txt')
        pairs=self._read_line(path)
        return self._create_unit(pairs,"train")
    
    @overrides
    def get_dev_units(self,data_dir):
        path=os.path.join(data_dir,'dev.txt')
        pairs=self._read_line(path)
        return self._create_unit(pairs,"dev")
    
    @overrides
    def get_labels(self):
        return args.LABELS
        
    def _create_unit(self,pairs,set_type):
        units=[]
        #sent_list=['李','华','在','北','京']
        for i, (sent,tags) in enumerate(pairs):
            guid="{}-{}".format(set_type,str(i))
            text_a=''.join(sent)
            assert len(tags)==len(text_a)
            unit=InputUnit(guid=guid,text_a=text_a,text_b=None,label=tags)
            units.append(unit)
        return units

class MRCDataProcessor(DataProcessor):
    def __init__(self):
        super(MRCDataProcessor,self).__init__()
    
    @overrides
    def get_train_units(self,data_dir):
        pass
    
    @overrides
    def get_dev_units(self,data_dir):
        pass
    
    @overrides
    def get_labels(self):
        return args.MRC_LABEL

                
                