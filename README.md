# MRC_NER
>In this project, I want to accumulate the NLP model I have learned Not Only MRC_NER. 
>
>Additionally, I try to build them up in one package,
>
>which means when I want to add and test a new model, I can avoid writing the repeated code.
>
>QAQ



## MRC_NER

According to [A Unifed MRC Frameword for Named Entity Recognition](https://arxiv.org/abs/1910.11476), It is good idea to solve the nested named entity recognition task. I reproduce this paper in another way. In this paper, the author designed 2 binary-classifier. One is to predict start index, another is to predict end Index. And In the loss function, the author proposed a Start-End Matching Loss. Personally speaking, it sounds very complex. In my code, I just simplify the model as a classifier which can classify one word into 3 categories, ['B','I','O']. As for the loss function, I use crossEntropy.

Because it is difficult to find the nested NER dataset (the famous datasets are too expensive 233), I just use flat ner dataset.



**How to use my code**

**Step 1** 

Preprocess the data, transform the raw data into MRC_training data

```shell
python -m src.data.preprocess
```



**Step 2**

training , validation , testing

```shell
python -m src.main
```



**output**

|               |   B    |   I    |   O    | macro  | micro  |
| ------------- | :----: | :----: | :----: | :----: | ------ |
| **acc**       | 0.9971 | 0.9928 | 0.9911 |        |        |
| **precision** | 0.9470 | 0.9346 | 0.9986 | 0.9601 | 0.9905 |
| **recall**    | 0.9817 | 0.9885 | 0.9911 | 0.9871 | 0.9905 |
| **f1_score**  | 0.9640 | 0.9608 | 0.9948 | 0.9734 | 0.9905 |



**To do list:**

- [ ] A implementation or a method to directly get the output if people give a sentence. You know it !

## Transformer

According to [Attention is all you need](https://arxiv.org/abs/1706.03762), for the code, I borrow the [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html#attention).

I finished the model architecture. 

**To do List**:

- [ ] Finding Dataset
- [ ] Dataset Preprocessing
- [ ] Model Wrapper for training,validation,testing
- [ ] running model function





## Package Building

**To do List:**

- [ ] Log Package
- [ ] Load and Save the torch model instead of the whole wrapper

