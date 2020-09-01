## Description of WebNLG
Because I try to reproduce [the Hierarchical Binary Tagging Framewor](https://arxiv.org/abs/1909.03227), I use the WebNLG as the Dataset to validate my Model.
I following the [data generate guide](https://github.com/weizhepei/CasRel/tree/master/data/WebNLG/raw_WebNLG) shared by the paper authors to get these jsons.

Every data item is a dict, in which 
Keyword **sentText** can get the raw sentence text
Keyword **relationMentions** can get a list

Every item of this list is a dict in which have three keys
- **em1Text**
- **em2Text**
- **label**

Under the triple directory, we can directly use thses three json file.
each of these file is consisted of 
``` json
{
    "Text":sentence
    "triple_list":[
        [o,r,s]
        [o,r,s]
    ]
}
```

I generate this following the data generate guide

---
**The Data Check**
|      |  Nums  |    Num of relation type  |   Average lenth    | Max_len|
| ---- | ---- | ---- | ----| ----|
| train | 5019 | 170 | 127.3 |452|
| dev| 500 | 110 | 121.6 |336|
| train | 703 | 125 | 128.3 |381|
| train | 6222 | 171 | 126.9 | 452|





