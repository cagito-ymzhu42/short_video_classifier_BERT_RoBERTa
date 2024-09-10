# -*- coding: utf-8 -*-

import jieba

from base._sentence_cls_dataset import SentenceClassificationDataset

class SentenceCustomizedClsDataset(SentenceClassificationDataset):
    """
    用于bert下句子分类任务
    """
    def __init__(self, *args, **kwargs):
        super(SentenceCustomizedClsDataset, self).__init__(*args, **kwargs)

    def _convert_to_custmoized_ids(self):
        pass
