#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import torch
from typing import List, Tuple


class HistoryBuffer:
    """
    Track a series of scalar values and provide access to smoothed values over a
    window or the global average of the series.
    """

    def __init__(self, max_length: int = 1000000):
        """
        Args:
            max_length: maximal number of values that can be stored in the
                buffer. When the capacity of the buffer is exhausted, old
                values will be removed.
        """
        self._max_length: int = max_length
        self._data: List[Tuple[float, float]] = []  # (value, iteration) pairs
        self._count: int = 0
        self._global_avg: float = 0

    def update(self, value: float, iteration: float = None):
        """
        Add a new scalar value produced at certain iteration. If the length
        of the buffer exceeds self._max_length, the oldest element will be
        removed from the buffer.
        """
        if iteration is None:
            iteration = self._count
        if len(self._data) == self._max_length:
            self._data.pop(0)
        self._data.append((value, iteration))

        self._count += 1
        self._global_avg += (value - self._global_avg) / self._count

    def latest(self):
        """
        Return the latest scalar value added to the buffer.
        """
        return self._data[-1][0]

    def median(self, window_size: int):
        """
        Return the median of the latest `window_size` values in the buffer.
        """
        return np.median([x[0] for x in self._data[-window_size:]])

    def avg(self, window_size: int):
        """
        Return the mean of the latest `window_size` values in the buffer.
        """
        return np.mean([x[0] for x in self._data[-window_size:]])

    def global_avg(self):
        """
        Return the mean of all the elements in the buffer. Note that this
        includes those getting removed due to limited buffer storage.
        """
        return self._global_avg

    def values(self):
        """
        Returns:
            list[(number, iteration)]: content of the current buffer.
        """
        return self._data


@torch.no_grad()
class MemoryBank():
    def __init__(self, cfg):
        data_names = ["Expert1", "Expert2", "Expert3"]
        class_nums = cfg.DATASETS.CLASS
        self.MB = dict()
        for name, num in zip(data_names, class_nums):
            self.MB[name] = dict()
            for i in range(num):
                self.MB[name][i] = dict()
                self.MB[name][i]["style"] = torch.zeros(2)
                torch.nn.init.normal_(self.MB[name][i]["style"], mean=0, std=1)
            #    self.MB[name][i]["feature"] = None
        self.empty = True
    
    def update(self, data_name, targets, feature, lmda=0.5):
        m_ = torch.mean(feature, dim=1).detach()
        n_ = torch.std(feature, dim=1).detach()
        if self.empty:
            for target, m, n, feat in zip(targets, m_, n_, feature):
                target = int(target)
            #    self.MB[data_name][target]["feature"] = feat
                self.MB[data_name][target]["style"] = torch.tensor([m,n])
            self.empty = False
        else:
            for target, m, n, feat in zip(targets, m_, n_, feature):
                target = int(target)
            #    self.MB[data_name][target]["feature"] = lmda * self.MB[data_name][target]["feature"]  + (1-lmda) * feat
                self.MB[data_name][target]["style"][0] = lmda * self.MB[data_name][target]["style"][0]  + (1-lmda) * m
                self.MB[data_name][target]["style"][1] = lmda * self.MB[data_name][target]["style"][1]  + (1-lmda) * n

    def values(self, data_name, targets, info="style"):
        if info == "style":
            styles = []
            for target in targets:
                styles.append(self.MB[data_name][target]["style"])
            return torch.tensor(styles).detach()
        
    def expert_styles(self, data_name):
        styles = []
        for v in self.MB[data_name].values():
            styles.append(v["style"])
        return torch.stack(styles, 0)

"""data_names = ["Expert1", "Expert2", "Expert3"]
class_nums = [2, 4, 6]
MB = dict()
for name, num in zip(data_names, class_nums):
        MB[name] = dict()
        for i in range(num):
            MB[name][i] = dict()
            MB[name][i]["style"] = None
            MB[name][i]["feature"] = None

data_name = "Expert3"
style = [[1,2], [3,4]]
targets = [0,1]
MB[name][targets]['style'] = 1

print(MB)"""

