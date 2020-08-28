import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from src.config.TrainingConfig import TrainingConfig

class BaseWrapper(object):
    def __init__(self):
        self.device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.lr=TrainingConfig.lr
        self.batch_size=TrainingConfig.batch_size
        self.epoches=TrainingConfig.epoches
        self.print_step=TrainingConfig.print_step
        
        self.step=0
        self._best_val_loss=1e18
        self.best_model=None

        #需要重写的
        self.model=None
        self.optimizer=None
    
    def train(self,train_data_loader,dev_data_loader,**kwargs):
        for e in range(self.epoches):
            self.step=0
            losses=0.
            self.model.train()
            for data in train_data_loader:
                self.step+=1
                losses=self._step_train(data,**kwargs)
                if self.step % self.print_step==0:
                    total_step=train_data_loader.dataset.__len__()//self.batch_size +1
                    print("Epoch {}, step/total_step: {}/{} Average Loss for one batch:{:.4f}".format(e+1,self.step,total_step,losses/self.print_step))
                    losses=0.
            
            #增加这个判定，是为了在第一个Validation中best_model没有初始化，回报错
            #理论上来说，在训练了一个epoch之后，如果模型进行了学习，best_model肯定会进行更新
            if e==0:
                self.best_model=self.model
            
            val_loss=self.validate(dev_data_loader,**kwargs)
            print("Validation:\t Epoch {}, Val Loss:{:.4f}".format(e+1,val_loss))
                
    def validate(self,dev_data_loader,**kwargs):
        self.model.eval()
        with torch.no_grad():
            val_losses=0.
            val_step=0
            ans=None
            for data in dev_data_loader:
                val_step+=1
                loss=self._cal_loss(data,**kwargs)
                loss=loss.item()
                val_losses+=loss
                #return的 loss 不需要backward() 取了item
        
                #计算eval unit
                units=self._eval_unit(data,**kwargs)
                ans=[i+j for i,j in zip(ans,units)] if ans!=None else units
                
        val_loss=val_losses/val_step
        #当模型在Validation Dataset上损失下降后，更新best_model 同时打印评价指标
        if val_loss < self._best_val_loss:
            self.best_model=deepcopy(self.model)
            self._best_val_loss=val_loss

            print('Upgrade Model and Save Model')
            print('-'*15+'In validation dataset, metrices'+'-'*15)
            #打印每个多分类中每一个分类的评价指标
            for unit in ans:
                unit.ptr()
            print('-'*61)

            
        return val_loss/self.batch_size

    def _step_train(self,batch_data,**kwargs):
        loss=self._cal_loss(batch_data,**kwargs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    #该method 只用于评测模型的效果，不评测在特定任务下的效果
    #返回一个自定义的unit list
    def test(self,test_data_loader,**kwargs):
        self.best_model.eval()
        total_step=test_data_loader.dataset.__len__()//self.batch_size+1
        ans=None
        with torch.no_grad():
            for step,batch_data in enumerate(test_data_loader):
                units=self._eval_unit(batch_data,**kwargs)
                ans=[i+j for i,j in zip(ans,units)] if ans!=None else units
                if step % self.print_step==0:
                    print('-'*15+'Step 112'+'-'*15);print('-'*60)
                    print('step/total_step:{}/{}'.format(step,total_step))
                    for unit in ans:
                        unit.ptr()
        return ans

    def _cal_loss(self,batch_data,**kwargs):
        raise NotImplementedError()

    def _eval_unit(self,batch_data,**kwargs):
        raise NotImplementedError()

    

        