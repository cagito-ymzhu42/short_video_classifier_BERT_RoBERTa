import time
from pretrained_pipeline.dataset.sentence_cls_bert_dataset import SentenceBertClsDataset
from pretrained_pipeline.processor.tokenizer.nezha import SentenceTokenizer
from pretrained_pipeline.factory.task.cls_task.sentence_cls_task import SentenceCLSTask, logging
from pretrained_pipeline.model.text_cls.nezha_cls_model import NezhaClsModel, NezhaAttClsModel
from pretrained_pipeline.model.text_cls.bert_model import BertClsModel, BertAttClsModel
from pretrained_pipeline.nn.nezha.configuration import NeZhaConfig

from pretrained_pipeline.factory.untils.attack import FGM, PGD
from pretrained_pipeline.factory.untils.tools import seed_torch, split_dataset
from pretrained_pipeline.factory.untils.opt import get_default_bert_optimizer

from transformers import AdamW
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Config:
    seed = 42
    max_sen_len = 64
    train_batch_size = 8
    val_batch_size = 8
    bert_pretrained_name = 'model/chinese-roberta-wwm-ext'
    pre_model_type = None
    multi_gpu = False
    params_path = './'
    n_classes = 19
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


fold_categories = ['保险', '储蓄', '其它', '创业', '基金', '外汇', '房产', '数字货币', '泛财经', '留学', '移民', '税务', '美女',
                   '股票-其他', '股票-基本面分析', '股票-技术面分析', '股票-行业分析', '财经段子', '鸡汤']

config = Config()
config.pre_model_type = config.bert_pretrained_name.split('/')[-1]
seed_torch(config.seed)

tokenizer = SentenceTokenizer(config.bert_pretrained_name, config.max_sen_len)
model = BertClsModel(config)
model.load(config.trained_model_path, device)
optimizer = get_default_bert_optimizer(model, lr=2e-5)
loss_func = nn.CrossEntropyLoss()
task = SentenceCLSTask(model, optimizer, loss_func, config)

# test_df_path = 'predict_简体中文.xlsx'
# predicted_path = 'predict_简体中文_roberta_0525.xlsx'
test_df_path = 'data/test_dataset_ch.xlsx'
predicted_path = 'results/test_dataset_roberta_0531.xlsx'
test_df = pd.read_excel(test_df_path)
test_df = test_df.sample(frac=0.002)
test_df = test_df.reset_index(drop=True)
# test_df = test_df[["text", "label", "备注"]]

start_time = time.time()
test_dataset = SentenceBertClsDataset(test_df, categories=fold_categories, is_train=False)
test_dataset.convert_to_ids(tokenizer)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.train_batch_size,
    shuffle=False,
    num_workers=config.num_workers,
)

predictions = []
pred_scores = []
for d in test_loader:
    input_ids = d["input_ids"].to(task.device)
    attention_mask = d["attention_mask"].to(task.device)

    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask)
    try:
        outputs = torch.softmax(outputs, dim=1)
        scores, preds = torch.max(outputs, dim=1)
        predictions.extend(preds.cpu().detach().numpy())
        pred_scores.extend(scores.cpu().detach().numpy())
    except Exception as e:
        print(input_ids)
        print(attention_mask)
        outputs = torch.softmax(outputs, dim=0)
        score, pred = torch.max(outputs, dim=0)
        predictions.append(int(pred))
        pred_scores.append(float(score))

time_consume = round(time.time() - start_time, 4)
print("推理" + str(len(test_df)) + "条数据，共耗时" + str(time_consume) + "秒")

predicted_labels = [test_dataset.id2cat[p] for p in predictions]
pred_scores = [round(float(t), 3) for t in pred_scores]
# labels = list(test_df["label"])
# is_same = [int(labels[i] == predicted_labels[i]) for i in range(len(predicted_labels))]
test_df['predicted_label'] = predicted_labels
test_df['predicted_score'] = pred_scores
# test_df['is_same'] = is_same
test_df.to_excel(predicted_path, index=False)
