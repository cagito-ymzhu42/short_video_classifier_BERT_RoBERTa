# -*- coding: utf-8 -*-

from pretrained_pipeline.nn.nezha.modeling import NeZhaModel
from pretrained_pipeline.nn.base.basemodel import BasicModel
from pretrained_pipeline.nn.layer.global_pointer_block import GlobalPointer
from pretrained_pipeline.factory.untils.tools import initial_parameter


class GlobalPointerNezha(BasicModel):
    """
    GlobalPointer + nezha 的命名实体模型

    Args:
        config: 模型的配置对象
        bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练

    Reference:
        [1] https://www.kexue.fm/archives/8373
    """
    def __init__(self, config, encoder_trained=True, head_size=64):
        super(GlobalPointerNezha, self).__init__()
        self.num_labels = config.num_labels
        self.bert = NeZhaModel.from_pretrained(config.bert_pretrained_name)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.global_pointer = GlobalPointer(
            self.num_labels,
            head_size,
            config.hidden_size
        )

        initial_parameter([self.global_pointer])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        encoder_out, pooled_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        print('encoder_out', encoder_out.shape)
        logits = self.global_pointer(encoder_out, mask=attention_mask)

        return logits