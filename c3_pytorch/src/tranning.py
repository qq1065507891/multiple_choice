import torch
import time
import os
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn import metrics
from transformers import BertTokenizer

from src.utils.process import Processor, process_data
from src.utils.utils import get_time_idf, make_seed, log
from src.models.bert_mutiple_choice import MultipleChoice
from src.utils.dataset import collate_fn


def traninng(config):
    """
    导入数据，开始训练
    :return:
    """
    make_seed(1001)
    log.info('开始预处理数据')
    start_time = time.time()

    log.info('*****************预处理数据集*********************')
    tokenizer = BertTokenizer.from_pretrained(config.bert_base_path)
    process = Processor(config)
    label_list = process.get_labels()
    train_examples = process.get_train_examples()
    dev_examples = process.get_dev_examples()
    train_data = process_data(config.out_path, config, train_examples,
                              tokenizer, config.max_len, label_list, name='train')

    dev_data = process_data(config.out_path, config, dev_examples,
                            tokenizer, config.max_len, label_list, name='dev')

    end_time = get_time_idf(start_time)
    log.info(f'加载数据完成， 用时： {end_time}')

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dataset=dev_data,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn
    )

    if config.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', config.gpu_id)
    else:
        device = torch.device('cpu')

    model = MultipleChoice(config).to(device)
    # model = BertForMultipleChoice.from_pretrained(config.bert_base_path).to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # 优化的参数
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_deacy': 0.0}
    ]

    model.train()

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                           factor=config.decay_rate,
                                                           patience=config.decay_patience)

    if os.path.exists(config.model_path):
        model.load_state_dict(torch.load(config.model_path))

    train_epoch(config, device, train_dataloader, dev_dataloader, model, loss_fn, optimizer, scheduler)


def train_epoch(config, device, train_dataloader, dev_dataloader, model, loss_fn, optimizer, scheduler=None):
    model.train()

    total_batch = 0  # 记录进行多少batch
    dev_best_loss = float('inf')  # 记录上次最好的验证集loss
    last_improve = 0  # 记录上次提升的batch
    flag = False  # 停止位的标志, 是否很久没提升

    start_time = time.time()

    for epoch in range(config.epochs):
        log.info('Epoch [{}/{}]'.format(epoch + 1, config.epochs))
        for i, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            *x, y = [data.to(device) for data in batch]
            # input_ids = x[0].to(device)
            # attention_mask = x[1].to(device)
            # y = y.to(device)
            outputs = model(x)

            model.zero_grad()

            loss = loss_fn(outputs, y)

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(models.parameters(), 1.0)
            optimizer.step()

            if total_batch % 100 == 0:  # 每训练50次输出在训练集和验证集上的效果
                true = y.data.cpu()
                output = torch.softmax(outputs, dim=-1)
                predict = torch.argmax(output.data, dim=-1).cpu()
                # predict = torch.max(output.data, 1)[1].cpu()
                score = metrics.accuracy_score(true, predict)

                dev_acc, dev_loss = evaluate(config, model, dev_dataloader, device)
                if total_batch > 20000:
                    scheduler.step(dev_loss)

                if dev_best_loss > dev_loss:
                    dev_best_loss = dev_loss

                    torch.save(model.state_dict(), config.model_path)
                    improve = '+'
                    last_improve = total_batch
                else:
                    improve = '-'
                time_idf = get_time_idf(start_time)
                msg = 'Iter:{0:>6}, Train Loss:{1:>5.6}, Train ACC:{2:>6.2%}, ' \
                      'Val Loss:{3:>5.6}, Val ACC:{4:>6.2%}, Time:{5}  {6}'
                log.info(msg.format(total_batch, loss.item(), score, dev_loss, dev_acc, time_idf, improve))
                model.train()

            total_batch = total_batch + 1

            if total_batch - last_improve > config.require_improvement:
                # 在验证集上loss超过1000batch没有下降, 结束训练
                log.info('在验证集上loss超过10000次训练没有下降, 结束训练')
                flag = True
                break

        if flag:
            break


def evaluate(config, model, dev_iter, device, test=False):
    """
    模型评估
    :param config:
    :param model:
    :param dev_iter:
    :param test:
    :return: acc, loss
    """
    model.eval()
    loss_total = 0
    predicts_all = np.array([], dtype=int)
    labels_all = np.array([], dtype='int')

    with torch.no_grad():
        for batch in dev_iter:
            *x, labels = [data.to(device) for data in batch]
            outputs = model(x)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            true = labels.data.cpu().numpy()
            outputs = torch.softmax(outputs, dim=-1)
            # predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict = torch.argmax(outputs.data, dim=-1).cpu().numpy()
            predicts_all = np.append(predicts_all, predict)
            labels_all = np.append(labels_all, true)

    acc = metrics.accuracy_score(labels_all, predicts_all)

    if test:
        report = metrics.classification_report(labels_all, predicts_all, target_names=config.class_name, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predicts_all)
        return acc, loss_total / len(dev_iter), report, confusion

    return acc, loss_total / len(dev_iter)



