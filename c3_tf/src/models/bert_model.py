import tensorflow as tf
from tensorflow.keras.layers import Lambda, Dense, Dropout
from tensorflow.keras import Model

import os

os.environ['TF_KERAS'] = '1'

from bert4keras.models import build_transformer_model
from bert4keras.backend import set_gelu


class MultillabelClassification(object):
    def __init__(self, config, last_activation='softmax', model_type='bert', dropout_rate=0):
        self.last_activation = last_activation
        self.model_type = model_type
        self.dropout_rate = dropout_rate
        self.config = config

        self.bert = build_transformer_model(
            config_path=config.config_path,
            checkpoint_path=config.checkpoint_path,
            model=self.model_type,
            return_keras_model=False,
        )
        self.classifier = Dense(units=1, activation=self.last_activation, name='output')

        self.drop_out = Dropout(self.config.drop_out)

    def build_model(self,):
        set_gelu('tanh')
        output = Lambda(lambda x: x[:, 0], name='CLS-token')(self.bert.model.output)
        pooled_output = self.drop_out(output)
        logits = self.classifier(pooled_output)
        reshape_layer = tf.reshape(logits, [-1, self.config.num_classes])
        model = Model(self.bert.input, reshape_layer, name=self.config.model_name)
        return model
