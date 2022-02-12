import paddle
import time
import os
import numpy as np


from paddlenlp.transformers import LinearDecayWithWarmup
from sklearn import metrics
from paddlenlp.transformers import BertTokenizer

from src.utils.process import Processor, process_data
from src.utils.utils import get_time_idf, make_seed, log
from src.models.bert_mutiple_choice import MultipleChoice


def traninng(config):
    """
    导入数据，开始训练
    :return:
    """
    make_seed(1001)
    log.info('开始预处理数据')
    start_time = time.time()

    log.info('*****************预处理数据集*********************')
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    process = Processor(config)
    label_list = process.get_labels()
    train_examples = process.get_train_examples()
    dev_examples = process.get_dev_examples()
    train_data = process_data(config.out_path, config, train_examples,
                              tokenizer, config.max_len, label_list, name='train')

    dev_data = process_data(config.out_path, config, dev_examples,
                            tokenizer, config.max_len, label_list, name='dev')

    train_dataloader = paddle.io.DataLoader(dataset=train_data,
                                            batch_size=config.batch_size,
                                            drop_last=True,
                                            num_workers=0)

    dev_dataloader = paddle.io.DataLoader(dataset=dev_data,
                                          batch_size=config.batch_size,
                                          drop_last=True,
                                          num_workers=0)

    end_time = get_time_idf(start_time)
    log.info(f'加载数据完成， 用时： {end_time}')

    model = MultipleChoice(config)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # 优化的参数
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_deacy': 0.0}
    ]

    max_grad_norm = 1.0
    num_training_steps = len(train_dataloader) * config.epochs
    lr_scheduler = LinearDecayWithWarmup(2e-5, num_training_steps, 0)

    grad_clip = paddle.nn.ClipGradByGlobalNorm(max_grad_norm)

    # 定义 Optimizer
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=0.01,
        apply_decay_param_fun=lambda x: x in optimizer_grouped_parameters,
        grad_clip=grad_clip)
    # 交叉熵损失
    criterion = paddle.nn.loss.CrossEntropyLoss()
    # 评估的时候采用准确率指标
    metric = paddle.metric.Accuracy()

    if os.path.exists(config.model_path):
        layer_state_dict = paddle.load(config.model_path)
        model.set_state_dict(layer_state_dict)

    train_epoch(config,  train_dataloader, dev_dataloader, model, criterion, optimizer,
                metric=metric, lr_scheduler=lr_scheduler)


def train_epoch(config, train_dataloader, dev_dataloader, model, loss_fn, optimizer, metric=None, lr_scheduler=None):
    model.train()

    total_batch = 0  # 记录进行多少batch
    dev_best_loss = float('inf')  # 记录上次最好的验证集loss
    last_improve = 0  # 记录上次提升的batch
    flag = False  # 停止位的标志, 是否很久没提升

    start_time = time.time()

    for epoch in range(config.epochs):
        log.info('Epoch [{}/{}]'.format(epoch + 1, config.epochs))
        for i, batch in enumerate(train_dataloader):
            *x, y = batch
            outputs = model(x)

            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if total_batch % 100 == 0:  # 每训练50次输出在训练集和验证集上的效果

                correct = metric.compute(outputs, y)
                metric.update(correct)
                acc = metric.accumulate()

                dev_acc, dev_loss = evaluate(config, model, dev_dataloader, metric)

                if dev_best_loss > dev_loss:
                    dev_best_loss = dev_loss

                    paddle.save(model.state_dict(), config.model_path)
                    improve = '+'
                    last_improve = total_batch
                else:
                    improve = '-'
                time_idf = get_time_idf(start_time)
                msg = 'Iter:{0:>6}, Train Loss:{1:>5.6}, Train ACC:{2:>6.2%}, ' \
                      'Val Loss:{3:>5.6}, Val ACC:{4:>6.2%}, Time:{5}  {6}'
                log.info(msg.format(total_batch, loss.item(), acc, str(dev_loss), dev_acc, time_idf, improve))
                model.train()

            total_batch = total_batch + 1

            if total_batch - last_improve > config.require_improvement:
                # 在验证集上loss超过1000batch没有下降, 结束训练
                log.info('在验证集上loss超过10000次训练没有下降, 结束训练')
                flag = True
                break

        if flag:
            break


@paddle.no_grad()
def evaluate(config, model, dev_iter, metric):
    """
    模型评估
    :param config:
    :param model:
    :param dev_iter:
    :param test:
    :return: acc, loss
    """
    model.eval()
    loss_total = []
    acc = []
    loss_fn = paddle.nn.loss.CrossEntropyLoss()

    for batch in dev_iter:
        *x, labels = batch
        outputs = model(x)
        loss = loss_fn(outputs, labels)
        loss_total.append(loss)

        correct = metric.compute(outputs, labels)
        metric.update(correct)

    acc = metric.accumulate()

    return acc, np.mean(loss_total)

