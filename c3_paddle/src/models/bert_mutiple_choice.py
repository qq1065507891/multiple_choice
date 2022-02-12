import paddle.nn as nn

from paddlenlp.transformers import BertModel


class MultipleChoice(nn.Layer):
    def __init__(self, config):
        super(MultipleChoice, self).__init__()
        self.config = config

        self.model_name = config.model_name

        self.bert = BertModel.from_pretrained(config.MODEL_NAME)

        self.classifier = nn.Linear(self.bert.config["hidden_size"], 1)

        self.drop_out = nn.Dropout(self.config.drop_out)

    def forward(self, inputs):
        """
        :param inputs: 【input_ids, attention_mask, attention_mask】
        :return:
        """
        input_ids, token_type_ids = inputs

        input_ids = input_ids.reshape(shape=(
            -1, input_ids.shape[-1]))  # flat_input_ids: [bs*num_choice,seq_l]

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(shape=(
                -1, token_type_ids.shape[-1]))

        _, pooled_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids
        )
        pooled_output = self.drop_out(pooled_output)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(shape=(-1, self.config.num_choices))

        return reshaped_logits




