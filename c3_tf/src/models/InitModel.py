import tensorflow as tf

from src.models.bert_model import MultillabelClassification


def init_model(config):
    """
    初始化模型
    :param config:
    :return:
    """
    model = MultillabelClassification(config, last_activation='softmax', dropout_rate=config.drop_out).build_model()

    optim = tf.keras.optimizers.Adam(config.learning_rate)
    loss = tf.keras.losses.sparse_categorical_crossentropy
    metrics = tf.keras.metrics.sparse_categorical_accuracy
    # metrics = ['acc']

    model.compile(optimizer=optim, loss=loss, metrics=metrics)

    return model


def load_model(config):
    """
    加载模型
    :param config:
    :return:
    """
    model = init_model(config)
    model.load_weights(config.model_path)
    return model
