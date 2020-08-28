from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from src.utils.common import flatten_lists
from tabulate import tabulate
class Eval_Unit(object):
    def __init__(self,tp,fp,fn,tn,label):
        super(Eval_Unit,self).__init__()
        self.id=label
        self.d={'tp':tp,'fp':fp,'fn':fn,'tn':tn}
        self.accuracy=self.cal_accuracy(tp,fp,fn,tn)
        self.precision=self.cal_precision(tp,fp,fn,tn)
        self.recall=self.cal_recall(tp,fp,fn,tn)
        self.f1_score=self.cal_f1_score(tp,fp,fn,tn)

    def __getattr__(self,name):
        return self[name] if name in self.__dict__ else self.d[name]

    def __add__(self,other):
        if isinstance(other,Eval_Unit) and self.id==other.id:
            tp=self.tp+other.tp
            fp=self.fp+other.fp
            fn=self.fn+other.fn
            tn=self.tn+other.tn
            return Eval_Unit(tp,fp,fn,tn,self.id)
    
    def todict(self):
        return {"acc":self.accuracy,"prec":self.precision,"recall":self.recall,"f1_score":self.f1_score}
    
    def ptr(self):
        print("Label ids: {} \t accuracy:{} \t precision:{} \t recall:{} "\
            .format(self.id,self.accuracy,self.precision,self.recall))
    @classmethod
    def cal_accuracy(cls,tp:int,fp:int,fn:int,tn:int)->float:
        return float(tp+tn)/(tp+tn+fp+fn)
    
    @classmethod
    def cal_precision(cls,tp:int,fp:int,fn:int,tn:int)->float:
        return float(tp)/(tp+fp) if tp+fp!=0 else 0.
    
    @classmethod
    def cal_recall(cls,tp:int,fp:int,fn:int,tn:int)->float:
        return float(tp)/(tp+fn) if tp+fn!=0 else 0.
    
    @classmethod
    def cal_f1_score(cls,tp:int,fp:int,fn:int,tn:int)->float:
        p=cls.cal_precision(tp,fp,fn,tn)
        r=cls.cal_recall(tp,fp,fn,tn)
        return 2*p*r/(r+p) if r+p !=0 else 0.


'''
将多分类混淆矩阵转化成 Unit_list list
生成器
'''

def confusion_matrix_to_units(pred,tag,ids2labels):
    classes=list(ids2labels.keys())
    matrix=confusion_matrix(pred,tag,classes)
    TP=np.diag(matrix)
    FP=matrix.sum(axis=1)-TP
    FN=matrix.sum(axis=0)-TP
    TN=matrix.sum()-TP-FN-FP
    units=[]
    for i,cla in enumerate(classes):
        units.append(Eval_Unit(TP[i],FP[i],FN[i],TN[i],ids2labels.get(cla)))
    return units



'''
将Eval_unit  的list 转化成 pandas 中的 DataFrame
同时计算Macro 和 Micro 的 Precision Recall F1-score
'''
def convert_units_to_dataframe(units,ptr=True):
    d={}
    macro=_evaluate_multiclass(units,"macro")
    micro=_evaluate_multiclass(units,"micro")
    d=dict((unit.id,unit.todict()) for unit in units)
    d["macro"]=macro
    d["micro"]=micro
    df=pd.DataFrame(d)
    if ptr:
        print(tabulate(df,headers='keys',tablefmt='psql'))
    return df

def _evaluate_multiclass(units:list,type:str):
    assert type in ['macro','micro']
    if type=='macro':
        P=float(sum([unit.precision for unit in units]))/len(units)
        R=float(sum([unit.recall for unit in units]))/len(units)

    else:
        tp=float(sum([unit.tp for unit in units]))/len(units)
        fp=float(sum([unit.fp for unit in units]))/len(units)
        fn=float(sum([unit.fn for unit in units]))/len(units)
        P=tp/(tp+fp)
        R=tp/(tp+fn)
    f1=2*P*R/(P+R)

    return {"prec":P,"recall":R,"f1_score":f1}