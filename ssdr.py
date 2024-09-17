#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/9/17 12:27
# @Author  : renyumeng
# @File    : ssdr.py
# @Software: PyCharm
import numpy as np
import torch


def generate_constraints(n_samples: int, constraints_num: int, must_link_ratio: float = 0.2, is_labeled=False,
                         y: torch.Tensor = None) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]]]:
    """
    生成约束
    :param n_samples: 样本数量
    :param y: 标签
    :param is_labeled: 数据集带标签
    :param constraints_num:约束数量
    :param must_link_ratio: ml数据比例
    :return: must-link,cannot-link下标序列
    """
    must_link = []
    cannot_link = []
    for i in range(constraints_num):
        i, j = np.random.choice(n_samples, 2, replace=False)
        if not is_labeled:
            # TODO 不带标签数据取百分比作为约束
            if np.random.rand() < must_link_ratio:
                must_link.append((i, j))
            else:
                cannot_link.append((i, j))
        else:
            must_link.append((i, j)) if y[i] == y[j] else cannot_link.append((i, j))
    return must_link, cannot_link


def ssdr(X: torch.Tensor, must_link: list[tuple[int, int]], cannot_link: list[tuple[int, int]], d=2, alpha=1, beta=20,
         have_unable=True) -> torch.Tensor:
    """
    计算线性变化的矩阵W
    :param X: 原始数据
    :param must_link: 约束
    :param cannot_link: 约束
    :param d: 降到多少维
    :param alpha: 超参数用于控制勿连约束的权重
    :param beta: 超参数用于控制必连约束的权重
    :param have_unable: 存在没有标签的数据
    :return:
    """
    n = X.shape[0]
    S = torch.full((n, n), 1 / (n ** 2)) if have_unable else torch.zeros((n, n))

    for i, j in must_link:
        S[i, j] = -beta / len(must_link)
        S[j, i] = S[i, j]

    for i, j in cannot_link:
        S[i, j] = alpha / len(cannot_link)
        S[j, i] = S[i, j]

    D = torch.diag(S.sum(dim=1))
    L = D - S

    XLXT = torch.matmul(torch.matmul(X.T, L), X)
    eigvals, eigvecs = torch.linalg.eigh(XLXT)
    sort_idx = torch.argsort(eigvals, descending=True)
    last_eigvecs = eigvecs[:, sort_idx]
    W = last_eigvecs[:, :d]
    return W


if __name__ == '__main__':
    n_samples, n_features = 100, 20
    X = torch.randn((n_samples, n_features))
    must_link, cannot_link = generate_constraints(n_samples, constraints_num=50, must_link_ratio=0.5)

    W = ssdr(X.clone().detach(), must_link, cannot_link, d=2)
    X_reduced = torch.matmul(X.clone().detach(), W).numpy()

    print("降维后的数据为:", X_reduced[:20, :])
