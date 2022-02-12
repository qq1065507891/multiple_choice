import os

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(curPath)[0])[0]


class Config(object):
    """
    配置类
    """

    model_name = 'MultipleChoice'

    log_folder_path = os.path.join(root_path, 'log')
    log_path = os.path.join(log_folder_path, 'log.txt')

    bert_base_path = os.path.join(root_path, 'bert-base-chinese')

    data_path = os.path.join(root_path, 'data/c3')
    out_path = os.path.join(data_path, 'out')
    train_path = os.path.join(out_path, 'train.pkl')
    dev_path = os.path.join(out_path, 'dev.pkl')

    test_path = os.path.join(out_path, 'test.pkl')

    model_path = os.path.join(root_path, 'models/Multiple.bin')

    train = True
    use_gpu = True

    decay_rate = 0.3
    decay_patience = 10

    gpu_id = 0

    max_len = 400

    batch_size = 2
    epochs = 100
    require_improvement = 5000

    num_choices = 4
    hidden_size = 128
    drop_out = 0.2
    learning_rate = 2e-5


config = Config()
