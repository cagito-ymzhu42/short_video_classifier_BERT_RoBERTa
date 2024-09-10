import os
import time

import pandas as pd

import torch
from torch import nn

from pretrained_pipeline.processor.tokenizer.nezha import SentenceTokenizer
from pretrained_pipeline.factory.task.cls_task.sentence_cls_task import SentenceCLSTask, logging
from pretrained_pipeline.model.text_cls.bert_model import BertClsModel, BertAttClsModel

from pretrained_pipeline.factory.untils.tools import seed_torch
from pretrained_pipeline.factory.untils.opt import get_default_bert_optimizer

os.environ['OPENBLAS_NUM_THREADS'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Config:
    seed = 42
    max_sen_len = 128
    train_batch_size = 8
    val_batch_size = 8
    bert_pretrained_name = 'model/bert-base-uncased'
    pre_model_type = None
    multi_gpu = False
    params_path = './'
    n_classes = 15
    cuda_device = 0

    trained_model_path = './checkpoints/bert-base-uncased_en_0607.pth'
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


categories = ['保险', '储蓄', '其它', '创业', '基金', '房产', '数字货币', '泛财经', '税务', '美女', '股票-其他',
              '股票-基本面分析', '股票-技术面分析', '股票-行业分析', '鸡汤']

id2cat = dict(zip(range(len(categories)), categories))

config = Config()
config.pre_model_type = config.bert_pretrained_name.split('/')[-1]
seed_torch(config.seed)

tokenizer = SentenceTokenizer(config.bert_pretrained_name, config.max_sen_len)
model = BertClsModel(config)
model.load(config.trained_model_path, device)
optimizer = get_default_bert_optimizer(model, lr=2e-5)
loss_func = nn.CrossEntropyLoss()
task = SentenceCLSTask(model, optimizer, loss_func, config)


def sent2vec(sentence):
    sentence = sentence.lower().strip()
    encoding = tokenizer.sequence_to_ids(sentence)
    feature = {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'ori_text': sentence
    }
    input_ids = feature["input_ids"].to(task.device)
    attention_mask = feature["attention_mask"].to(task.device)
    input_ids = input_ids[None, :]
    attention_mask = attention_mask[None, :]
    vects, outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    outputs = torch.softmax(outputs, dim=0)
    score, pred = torch.max(outputs, dim=0)
    pred_label = id2cat[int(pred)]
    score = round(float(score), 3)
    return vects[1][0].detach().numpy(), pred_label, score


# sentence = "五条短线操盘策略，听懂点赞#财经 #干货分享 #财经知识"
# sentence = "1分鐘系列 - 看懂損益表 10個財務比率，算出公司內在價值。"
# sentence = "黃國英：我不相信nft ！"
# sentence = zhconv.convert(sentence, 'zh-cn')
# print(sent2vec(sentence))

# df = pd.read_excel("data/short_video_added-en-0628.xlsx")
df = pd.read_excel("data/item_info_en_0706.xlsx")

start_time = time.time()
with open("results/short_video_text_vector_en_0705.txt", "w", encoding="UTF-8") as f:
    for i in range(len(df)):
        id = df["index"][i]
        sent = df["video_text"][i]
        sent = str(sent)
        array, score, label = list(sent2vec(sent))
        f.write(str(id))
        f.write("\t" + str(label))
        f.write("\t" + str(score))
        for a in array:
            f.write("\t" + str(a))
        f.write("\n")
time_consume = round(time.time() - start_time, 4)
print("向量化" + str(len(df)) + "条数据，共耗时" + str(time_consume) + "秒")