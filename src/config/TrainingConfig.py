
class TrainingConfig():
    batch_size=64
    lr=0.001
    epoches=20
    print_step=15

class BertMRCTrainingConfig(TrainingConfig):
    batch_size=64
    lr=1e-5
    epoches=5

class TransformerConfig(TrainingConfig):
    pass


class HBTTrainingConfig(TrainingConfig):
    batch_size=32
    lr=1e-5
    