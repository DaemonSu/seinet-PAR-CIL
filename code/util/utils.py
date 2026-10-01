import torch

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from datetime import datetime
import logging
import matplotlib.pyplot as plt


def accuracy(preds, labels):
    return (preds == labels).float().mean().item()


# 文件：utils.py（补充 accuracy 函数）

import torch

def topaccuracy(output, target, topk=(1,)):
    """
    计算 top-k 准确率
    :param output: [B, num_classes]，模型输出 logits
    :param target: [B]，真实标签
    :param topk: tuple，支持多种 top-k 评估
    :return: list of top-k accuracy (%)
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # 获取 top-k 的预测类别
    _, pred = output.topk(maxk, 1, True, True)  # [B, maxk]
    pred = pred.t()  # [maxk, B]
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # [maxk, B]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Top-k 正确个数
        acc = correct_k.mul_(100.0 / batch_size)  # 百分比准确率
        res.append(acc.item())
    return res[0] if len(res) == 1 else res



def testAccuracy(preds, labels):
    np.set_printoptions(threshold=np.inf)
    pr=preds.cpu().numpy()
    # print("预测的分类是:", pr)  # 打印出来
    # print("实际设备编号是:", labels.cpu().numpy())  # 打印出
    correct = ((preds == labels) | ((preds == -1) & (labels > 10))).float()  # 新的条件
    accuracy = correct.mean().item()  # 计算准确率
    return accuracy

def set_seed(seed):
    """
    固定随机种子，保证可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(save_dir, name='train'):
    """
    保存日志文件
    """
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f'{name}.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def save_checkpoint(state, save_path):
    """
    保存模型与优化器状态
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)


def load_checkpoint(model, optimizer, ckpt_path, device):
    """
    加载模型与优化器状态
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint.get('epoch', 0)
    return model, optimizer, start_epoch


def calc_accuracy(y_true, y_pred):
    """
    计算分类精度
    """
    acc = metrics.accuracy_score(y_true, y_pred)
    return acc * 100.0


def calc_open_set_metrics(y_true, y_pred, y_score, threshold):
    """
    计算 Open-set 各类指标
    """
    known_mask = (y_true >= 0)
    pred_known_mask = (y_score >= threshold)

    # Open-set Detection Accuracy
    detect_acc = (known_mask == pred_known_mask).sum() / len(y_true)

    # Closed-set Classification Accuracy
    closed_acc = metrics.accuracy_score(
        y_true[known_mask & pred_known_mask],
        y_pred[known_mask & pred_known_mask]
    ) if (known_mask & pred_known_mask).sum() > 0 else 0.0

    return detect_acc * 100.0, closed_acc * 100.0


def adjust_openmax_threshold(score_list, target_rate=0.95):
    """
    自动调整 OpenMax 阈值
    """
    score_list = sorted(score_list)
    index = int(len(score_list) * (1 - target_rate))
    threshold = score_list[index]
    return threshold


def now():
    """
    获取当前时间字符串
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


import pickle
import os


def save_object(obj, filename):
    """
    保存对象到文件
    :param obj: 需要保存的对象
    :param filename: 保存路径，例如：'save/model.pkl'
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # 自动创建路径
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to: {filename}")


def load_object(filename):
    """
    从文件加载对象
    :param filename: 文件路径
    :return: 加载的对象
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file: {filename}")

    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from: {filename}")
    return obj


def analyze_logits_distribution(logits: torch.Tensor):
    # 转为 numpy 方便处理
    logits_np = logits.detach().cpu().numpy()

    np.set_printoptions(threshold=np.inf)
    # print("min_dist:", logits_np)  # 打印出来

    print("==== Logits 分布基本信息 ====")
    print(f"整体形状: {logits_np.shape}")
    print(f"全体最大值: {logits_np.max():.4f}")
    print(f"全体最小值: {logits_np.min():.4f}")
    print(f"均值: {logits_np.mean():.4f}")
    print(f"标准差: {logits_np.std():.4f}")

    num_classes = logits_np.shape[1]
    plt.figure(figsize=(16, 4))

    for i in range(num_classes):
        plt.subplot(1, num_classes, i + 1)
        plt.hist(logits_np[:, i], bins=30, alpha=0.7, color='skyblue')
        plt.title(f'Class {i}')
        plt.xlabel('Logit Value')
        plt.ylabel('Count')
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def compute_energy(logits, temperature=0.1):
    """
    Compute the energy score for a batch of logits.
    E(x) = -T * logsumexp(logits / T)
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


# 文件：utils.py（补充部分）

def adjust_lr(optimizer, epoch, config):
    """
    动态调整学习率，支持多步衰减策略。
    config 中需包含如下字段：
        - lr: 初始学习率
        - min_lr: 最小学习率
        - lr_decay_epochs: [epoch1, epoch2, ...]
        - lr_decay_rate: 衰减系数（如 0.1）
    """
    lr = config.lr
    decay_rate = config.lr_decay_rate
    decay_epochs = config.lr_decay_epochs
    min_lr = config.min_lr

    decay_steps = sum(epoch >= e for e in decay_epochs)
    new_lr = lr * (decay_rate ** decay_steps)
    new_lr = max(new_lr, min_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def adjust_lr_add(optimizer, epoch, config):
    """
    动态调整学习率，支持多步衰减策略。
    config 中需包含如下字段：
        - lr: 初始学习率
        - min_lr: 最小学习率
        - lr_decay_epochs: [epoch1, epoch2, ...]
        - lr_decay_rate: 衰减系数（如 0.1）
    """
    decay_rate = config.incre_lr_decay_rate
    decay_epochs = config.incre_lr_decay_epochs
    min_lr = config.min_lr

    decay_steps = sum(epoch >= e for e in decay_epochs)

    for param_group in optimizer.param_groups:
        base_lr = param_group.get('initial_lr', config.incre_lr)
        new_lr = base_lr * (decay_rate ** decay_steps)
        param_group['lr'] = max(new_lr, min_lr)

import numpy as np
from sklearn.metrics import roc_curve

def fpr_at_95_tpr(known_scores, unknown_scores):
    """
    known_scores: 已知样本得分 (numpy array)
    unknown_scores: 未知样本得分 (numpy array)
    """
    labels = np.concatenate([np.ones_like(known_scores), np.zeros_like(unknown_scores)])
    scores = np.concatenate([known_scores, unknown_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)

    # 找到最接近 95% 的 TPR
    idx = np.argmin(np.abs(tpr - 0.95))
    return fpr[idx]


import numpy as np

def expected_calibration_error(probs, labels, n_bins=15):
    """
    probs: (N, C) softmax 概率 (numpy array)
    labels: (N,) 真实标签 (numpy array)
    n_bins: 分桶数量
    """
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()




