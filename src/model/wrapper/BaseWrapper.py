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
                    print("Epoch {}, step/total_step: {}/{} Average Loss for one batch:{:.4f}".format(e,self.step,total_step,losses/self.print_step))
                    losses=0.

            val_loss=self.validate(dev_data_loader,**kwargs)
            print("Epoch {}, Val Loss:{:.4f}".format(e,val_loss))
                
    
    def validate(self,dev_data_loader,**kwargs):
        self.model.eval()
        with torch.no_grad():
            val_losses=0.
            val_step=0
            for data in dev_data_loader:
                val_step+=1
                loss=self._cal_loss(data,**kwargs)
                loss=loss.item()
                #return的 loss 不需要backward() 取了item
                val_losses+=loss
        val_loss=val_losses/val_step
        if val_loss < self._best_val_loss:
            print('Upgrade Model and Save Model')
            self.best_model=deepcopy(self.model)
            self._best_val_loss=val_loss
        return val_loss

    def _step_train(self,batch_data,**kwargs):
        loss=self._cal_loss(batch_data,**kwargs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _cal_loss(self,batch_data,**kwargs):
        raise NotImplementedError()

    def test(self,test_data_loader,**kwargs):
        raise NotImplementedError()

    

    

        