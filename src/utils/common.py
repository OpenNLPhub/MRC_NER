
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

    