import os


curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(curPath)[0])[0]


class Config(object):

    model_name = 'MultillabelClassification'
    model_type = 'bert'

    log_folder_path = os.path.join(root_path, 'log')
    log_path = os.path.join(log_folder_path, 'log.txt')

    config_path = os.path.join(root_path, 'bert_pretrain/bert_config.json')
    checkpoint_path = os.path.join(root_path, 'bert_pretrain/bert_model.ckpt')
    dict_path = os.path.join(root_path, 'bert_pretrain/vocab.txt')

    data_path = os.path.join(root_path, 'data')
    out_path = os.path.join(data_path, 'out')

    train_out_path = os.path.join(out_path, 'train.pkl')
    test_out_path = os.path.join(out_path, 'test.pkl')
    dev_out_path = os.path.join(out_path, 'dev.pkl')

    model_path = os.path.join(root_path, 'model/bert_f.h5')

    max_len = 400

    num_classes = 2
    learning_rate = 2e-5
    drop_out = 0.3
    batch_size = 4
    epochs = 100


config = Config()
