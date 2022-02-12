import time

import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
os.environ['TF_KERAS'] = '1'

from bert4keras.tokenizers import Tokenizer

from src.models.InitModel import load_model, init_model
from src.utils.process import Processor, process_data
from src.utils.utils import log, map_example_to_dict, training_curve, get_time_idf, set_seed
from src.utils.config import config


def training():
    set_seed(20001)
    tokenizer = Tokenizer(config.dict_path, do_lower_case=True)
    process = Processor(config)
    train_features = process.get_train_examples()
    test_features = process.get_test_examples()
    dev_features = process.get_dev_examples()
    label_list = process.get_labels()

    train_data = process_data(config.out_path, config, train_features, tokenizer, config.max_len,
                              label_list=label_list, name='train')
    test_data = process_data(config.out_path, config, test_features, tokenizer, config.max_len,
                             label_list=label_list, name='test')
    dev_data = process_data(config.out_path, config, dev_features, tokenizer, config.max_len,
                            label_list=label_list, name='dev')

    log.info('***********开始加载数据*********')
    start_time = time.time()

    train_iter = tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1], train_data[2]))\
        .shuffle(200).batch(config.batch_size, drop_remainder=True).map(map_example_to_dict)

    dev_iter = tf.data.Dataset.from_tensor_slices((dev_data[0], dev_data[1], dev_data[2]))\
        .shuffle(200).batch(config.batch_size, drop_remainder=True).map(map_example_to_dict)

    end_time = get_time_idf(start_time)

    log.info(f'数据加载完成, 用时:{end_time}, 训练数据: {len(list(train_iter))}, 验证数据： {len(list(dev_iter))}')

    log.info('*****开始加载模型******')

    if os.path.exists(config.model_path):
        log.info('*********已有模型，加载模型**********')
        model = load_model(config)
    else:
        log.info('*********初始化模型***********')
        model = init_model(config)
    model.summary()
    log.info('*********开始训练************')

    call_backs = [
        # EarlyStopping(monitor='val_categorical_accuracy', patience=5, verbose=1, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
                          epsilon=0.0001, cooldown=0, min_lr=0),
        ModelCheckpoint(config.model_path, monitor='val_sparse_categorical_accuracy', verbose=1,
                        save_weights_only=True, save_best_only=True, mode='max')
    ]

    start_time = time.time()

    history = model.fit(
        train_iter,
        epochs=config.epochs,
        validation_data=dev_iter,
        batch_size=config.batch_size,
        callbacks=call_backs
    )

    end_time = get_time_idf(start_time)
    log.info(f'训练完成，用时： {end_time}')
    training_curve(history.history['loss'], history.history['sparse_categorical_accuracy'],
                   history.history['val_loss'], history.history['val_sparse_categorical_accuracy'])
