from torch import nn
import pandas as pd

from pretrained_pipeline.dataset.sentence_cls_bert_dataset import SentenceBertClsDataset
from pretrained_pipeline.processor.tokenizer.nezha import SentenceTokenizer
from pretrained_pipeline.factory.task.cls_task.sentence_cls_task import SentenceCLSTask, logging
from pretrained_pipeline.model.text_cls.bert_model import BertClsModel, BertAttClsModel

from pretrained_pipeline.factory.untils.tools import seed_torch, split_dataset
from pretrained_pipeline.factory.untils.opt import get_default_bert_optimizer


class Config:
    seed = 42
    max_sen_len = 64
    train_batch_size = 8
    val_batch_size = 8
    bert_pretrained_name = 'model/chinese-roberta-wwm-ext'
    pre_model_type = None
    multi_gpu = False
    params_path = './'
    n_classes = 18
    cuda_device = 0

    trained_model_path = './checkpoints/roberta_0531.pth'
    # 训练模型参数
    num_workers = 0
    n_epoch = 8
    min_store_epoch = 2
    scheduler_type = 'get_linear_schedule_with_warmup'  # get_linear_schedule_with_warmup
    # trick 参数
    attack_func = None  # fgm  pgd
    pgd_k = 3
    is_use_rdrop = True
    alpha = 0.25  # ghmloss
    is_use_swa = False
    ema_decay = 0.99  # 0.995
    rdrop_ismean = False


config = Config()
config.pre_model_type = config.bert_pretrained_name.split('/')[-1]
seed_torch(config.seed)
tokenizer = SentenceTokenizer(config.bert_pretrained_name, config.max_sen_len)
fold_categories = None
# 加载训练用数据集
df = pd.read_excel('data/train_dataset_ch.xlsx')
train_df, dev_df = split_dataset(df)
train_dataset = SentenceBertClsDataset(train_df)
fold_categories = train_dataset.categories
print(fold_categories)
dev_dataset = SentenceBertClsDataset(dev_df, categories=fold_categories)

dev_dataset.convert_to_ids(tokenizer)
train_dataset.convert_to_ids(tokenizer)
config.n_classes = train_dataset.class_num

model = BertClsModel(config)
optimizer = get_default_bert_optimizer(model, lr=2e-5)
loss_func = nn.CrossEntropyLoss()
task = SentenceCLSTask(model, optimizer, loss_func, config)

task.fit(train_dataset, config, dev_dataset)

