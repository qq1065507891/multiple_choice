import os
import random
import paddle

from tqdm import tqdm
from paddle.io import TensorDataset

from src.utils.utils import log, read_file, save_pkl, ensure_dir, load_pkl


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, token_type_ids, attention_mask=None, label_id=None):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label_id = label_id


class Processor(object):
    def __init__(self, config):

        self.config = config
        self.D = [[], [], []]

        for sid in range(3):
            data = []
            if sid < 2:
                for subtask in ["d", "m"]:
                    data += read_file(config.data_path + '/' + subtask + "-" + ["train.json", "dev.json", "test.json"][sid])
            else:
                data += read_file(config.data_path + '/' + 'test1.0.json')
            if sid == 0:
                random.shuffle(data)

            if sid < 2:
                for i in range(len(data)):
                    for j in range(len(data[i][1])):
                        d = [
                            '\n'.join(data[i][0]).lower(),
                            data[i][1][j]["question"].lower()
                        ]
                        for k in range(len(data[i][1][j]["choice"])):
                            d += [data[i][1][j]["choice"][k].lower()]
                        for k in range(len(data[i][1][j]["choice"]), 4):
                            d += ['无效答案']  # 有些C3数据选项不足4个，添加[无效答案]能够有效增强模型收敛稳定性
                        d += [data[i][1][j]["answer"].lower()]
                        self.D[sid] += [d]
            else:
                for i in range(len(data)):
                    for j in range(len(data[i][1])):
                        d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
                        for k in range(len(data[i][1][j]["choice"])):
                            d += [data[i][1][j]["choice"][k].lower()]
                        for k in range(len(data[i][1][j]["choice"]), 4):
                            d += ['无效答案']  # 有些C3数据选项不足4个，添加[无效答案]能够有效增强模型收敛稳定性
                        d += [data[i][1][j]["choice"][0].lower()]
                        self.D[sid] += [d]

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self.D[0], "train")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(self.D[2], "test")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        ensure_dir(self.config.out_path)
        if set_type == 'train':
            path = self.config.train_path
        elif set_type == 'dev':
            path = self.config.dev_path
        else:
            path = self.config.test_path

        if os.path.exists(path):
            examples = load_pkl(path, set_type)
        else:
            examples = []
            # answer = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
            for (i, d) in enumerate(data):
                answer = -1
                # 这里data[i]有6个元素，0是context，1是问题，2~5是choice，6是答案
                for k in range(4):
                    if data[i][2 + k] == data[i][6]:
                        answer = str(k)

                label = answer
                for k in range(4):
                    guid = "%s-%s-%s" % (set_type, i, k)
                    text_a = data[i][0]
                    text_b = data[i][k + 2]
                    text_c = data[i][1]

                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, text_c=text_c))

            save_pkl(path, examples, set_type, use_bert=True)

        return examples


def convert_examples_to_features(examples, config, max_seq_length, tokenizer, label_list=None):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples))
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = [[]]
    for (ex_index, example) in enumerate(tqdm(examples)):
        data_encoding = tokenizer(example.text_a, example.text_b+example.text_c,
                                         max_seq_len=config.max_len, pad_to_max_seq_len=True)

        input_ids = data_encoding['input_ids']
        token_type_ids = data_encoding['token_type_ids']

        assert len(input_ids) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        label_id = label_map[example.label]
        # label_id = example.label

        if ex_index < 2:
            log.info("*** Example ***")
            log.info("guid: %s" % (example.guid))
            # logger.info("tokens: %s" % " ".join(
            #     [tokenization.printable_text(x) for x in tokens]))
            log.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            log.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            log.info("label: %s (id = %d)" % (example.label, label_id))

        features[-1].append(
            InputFeatures(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                label_id=label_id))
        if len(features[-1]) == config.num_choices:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features


def process_data(data_dir, config, examples,  tokenizer, max_seq_length, label_list=None, name='train'):

    feature_dir = os.path.join(data_dir, '{}_{}.pkl'.format(name, max_seq_length))
    if os.path.exists(feature_dir):
        process_features = load_pkl(feature_dir, name)
    else:
        process_features = convert_examples_to_features(examples, config, max_seq_length, tokenizer, label_list)
        save_pkl(feature_dir, process_features, name, use_bert=True)

    log.info("***** Running training *****")
    log.info("  Num examples = %d", len(process_features))

    input_ids = []
    token_type_ids = []

    label_id = []
    for f in tqdm(process_features):
        input_ids.append([])
        token_type_ids.append([])

        for i in range(config.num_choices):
            input_ids[-1].append(f[i].input_ids)
            token_type_ids[-1].append(f[i].token_type_ids)

        label_id.append(f[0].label_id)

    all_input_ids = paddle.to_tensor(input_ids, dtype='int64')
    all_token_type_ids = paddle.to_tensor(token_type_ids, dtype='int64')
    all_label_ids = paddle.to_tensor(label_id, dtype='int64')

    data = TensorDataset([all_input_ids, all_token_type_ids, all_label_ids])

    return data
