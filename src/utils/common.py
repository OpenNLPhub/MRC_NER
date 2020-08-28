
import pickle
def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


#用于将最后[B,L]的结果变成一维，用于计算结果
def flatten_lists(lists):
    ans=[]
    for l in lists:
        if type(l)==list:
            ans+=l
        else:
            ans.append(l)
    return ans;


'''
储存和加载模型
'''
def save_model(model,file_name):
    """Save the model"""
    print("Save the model")
    with open(file_name,"wb") as f:
        pickle.dump(model,f)


def load_model(file_name):
    """Load the model"""
    print("load the model")
    with open(file_name,"rb") as f:
        model=pickle.load(f)
    return model

