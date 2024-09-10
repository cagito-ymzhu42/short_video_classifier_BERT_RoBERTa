# coding=utf-8
import torch
from transformers import BertModel, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = BertTokenizer.from_pretrained("model/chinese-roberta-wwm-ext")
model = BertModel.from_pretrained("model/chinese-roberta-wwm-ext")
model.load_state_dict(torch.load("checkpoints/roberta_0531.pth", map_location=device))


def sent2vec(sentence):
    token_ids = tokenizer.encode(sentence)
    input_ids = torch.tensor(token_ids).unsqueeze(0)
    outputs = model(input_ids)
    sequence_output = outputs[0]
    pooled_output = outputs[1]
    return sequence_output, pooled_output


sentence = "新加坡家属准证可以申请留学吗？ #干货 #新加坡旅游 #新加坡疫情 #新加坡留学"
a, b = sent2vec(sentence)
print(a)
print(b)
