# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
import zhconv

from sanic import Sanic
from sanic.request import Request
from sanic import response
from sanic_cors import CORS

from conf.config import PretrainingConfigCH, PretrainingConfigEN

from pretrained_pipeline.processor.tokenizer.nezha import SentenceTokenizer
from pretrained_pipeline.factory.task.cls_task.sentence_cls_task import SentenceCLSTask, logging
from pretrained_pipeline.model.text_cls.bert_model import BertClsModel, BertAttClsModel

from pretrained_pipeline.factory.untils.tools import seed_torch
from pretrained_pipeline.factory.untils.opt import get_default_bert_optimizer

os.environ['OPENBLAS_NUM_THREADS'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 加载预训练模型-中文
pretraining_config_ch = PretrainingConfigCH()
pretraining_config_ch.pre_model_type = pretraining_config_ch.bert_pretrained_name.split('/')[-1]
seed_torch(pretraining_config_ch.seed)

tokenizer_ch = SentenceTokenizer(pretraining_config_ch.bert_pretrained_name, pretraining_config_ch.max_sen_len)
model_ch = BertClsModel(pretraining_config_ch)
model_ch.load(pretraining_config_ch.trained_model_path, device)
optimizer_ch = get_default_bert_optimizer(model_ch, lr=2e-5)
loss_func = nn.CrossEntropyLoss()
task_ch = SentenceCLSTask(model_ch, optimizer_ch, loss_func, pretraining_config_ch)

categories_ch = ['保险', '储蓄', '其它', '创业', '基金', '外汇', '房产', '数字货币', '泛财经', '留学', '移民', '税务', '美女',
                 '股票-其他', '股票-基本面分析', '股票-技术面分析', '股票-行业分析', '财经段子', '鸡汤']
id2cat_ch = dict(zip(range(len(categories_ch)), categories_ch))

pretraining_config_en = PretrainingConfigEN()
pretraining_config_en.pre_model_type = pretraining_config_en.bert_pretrained_name.split('/')[-1]
seed_torch(pretraining_config_en.seed)

tokenizer_en = SentenceTokenizer(pretraining_config_en.bert_pretrained_name, pretraining_config_en.max_sen_len)
model_en = BertClsModel(pretraining_config_en)
model_en.load(pretraining_config_en.trained_model_path, device)
optimizer_en = get_default_bert_optimizer(model_en, lr=2e-5)
loss_func = nn.CrossEntropyLoss()
task_en = SentenceCLSTask(model_en, optimizer_en, loss_func, pretraining_config_en)

categories_en = ['保险', '储蓄', '其它', '创业', '基金', '房产', '数字货币', '泛财经', '税务', '美女', '股票-其他',
                 '股票-基本面分析', '股票-技术面分析', '股票-行业分析', '鸡汤']
id2cat_en = dict(zip(range(len(categories_en)), categories_en))


class Worker:
    def __init__(self, video_language=1):
        self.video_language = video_language

    def predict_single_sentence_Roberta(self, sentence):
        if self.video_language == 2:
            sentence = zhconv.convert(sentence, 'zh-cn')
        # 英文处理预测流程
        if self.video_language == 3:
            sentence = sentence.lower().strip()
            encoding = tokenizer_en.sequence_to_ids(sentence)
            feature = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'ori_text': sentence
            }
            input_ids = feature["input_ids"].to(task_en.device)
            attention_mask = feature["attention_mask"].to(task_en.device)
            input_ids = input_ids[None, :]
            attention_mask = attention_mask[None, :]
            outputs = model_en(input_ids=input_ids, attention_mask=attention_mask)
            outputs = torch.softmax(outputs, dim=0)
            predict_score, pred = torch.max(outputs, dim=0)
            predict_label = id2cat_en[int(pred)]
            predict_score = round(float(predict_score), 3)
        elif self.video_language == 2 or self.video_language == 1:
            sentence = sentence.lower().strip()
            encoding = tokenizer_ch.sequence_to_ids(sentence)
            feature = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'ori_text': sentence
            }
            input_ids = feature["input_ids"].to(task_ch.device)
            attention_mask = feature["attention_mask"].to(task_ch.device)
            input_ids = input_ids[None, :]
            attention_mask = attention_mask[None, :]
            outputs = model_ch(input_ids=input_ids, attention_mask=attention_mask)
            outputs = torch.softmax(outputs, dim=0)
            predict_score, pred = torch.max(outputs, dim=0)
            predict_label = id2cat_ch[int(pred)]
            predict_score = round(float(predict_score), 3)
        return predict_label, predict_score


class ServiceConfig:
    service_host = "0.0.0.0"  # 顶层外部agent_server的host.
    service_port = 8085  # 顶层外部agent_server的port.
    bot_agent_server_url = "http://{}:{}".format(service_host, service_port)


class ModelService:
    def __init__(self, app_name="ShortVideoClassifier"):
        self.service_config = ServiceConfig()
        print(self.service_config.service_host, self.service_config.service_port)
        self.app = Sanic(app_name)
        CORS(self.app)

    def start_service(self):
        self.add_routes()
        self.app.run(self.service_config.service_host, self.service_config.service_port, workers=1)

    def add_routes(self):
        self.app.add_route(self.roberta_worker, "roberta", methods=["POST"])

    @staticmethod
    async def roberta_worker(request: Request):
        req_json = request.json
        video_language = req_json["video_language"]
        video_title = req_json['video_title'].strip()
        video_desc = req_json["video_desc"].strip()
        if video_title != video_desc:
            video_text = video_title + video_desc
        else:
            video_text = video_title
        clf_worker = Worker(video_language=video_language)
        result_label, result_score = clf_worker.predict_single_sentence_Roberta(video_text)
        print(result_label)
        print(result_score)
        resp_body = {
            "code": 200,
            "msg": "success",
            "label": result_label,
            "score": result_score
        }
        resp_json = response.json(resp_body, status=200)
        return resp_json


if __name__ == "__main__":
    server = ModelService()
    server.start_service()
