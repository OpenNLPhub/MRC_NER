
from src.data.dataloader import create_MRC_DataLoader,create_Bert_tokenizer
from src.model.wrapper.MRCBert_NER import MRCBert_NER
import src.config.args as args

def run_Bert_MRC_NER():
    tokenizer=create_Bert_tokenizer(True)
    train_data_loader=create_MRC_DataLoader('train',args.MRC_SOURCE_DATA,tokenizer)
    dev_data_loader=create_MRC_DataLoader('dev',args.MRC_SOURCE_DATA,tokenizer)
    vocab_size=len(tokenizer.__dict__.get('vocab'))
    labels=args.MRC_TAG
    model=MRCBert_NER(len(labels),True)
    model.train(train_data_loader,dev_data_loader,tokenizer=tokenizer)
    # model=MRCBert_NER()