import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class TestModel(BertPreTrainedModel):
    """
    基于BERT的文本分类模型
    """
    
    def __init__(self, config):
        """
        初始化模型
        
        Args:
            config: 模型配置
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 初始化权重
        self.init_weights()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        前向传播
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            token_type_ids: 词类型ID
            position_ids: 位置ID
            head_mask: 头掩码
            inputs_embeds: 输入嵌入
            labels: 标签
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典
            
        Returns:
            模型输出
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 获取BERT输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 获取池化输出
        pooled_output = outputs[1]
        
        # 应用dropout
        pooled_output = self.dropout(pooled_output)
        
        # 计算logits
        logits = self.classifier(pooled_output)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
            
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) 