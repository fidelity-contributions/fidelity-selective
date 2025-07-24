# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import NoReturn, Tuple

import pandas as pd
import numpy as np

from scipy.special import kl_div, rel_entr
from feature.base import _BaseSupervisedSelector, _BaseDispatcher
from feature.utils import Num, get_task_string, check_true


from tqdm import tqdm


class _KL_Divergence(_BaseSupervisedSelector, _BaseDispatcher):

    def __init__(self, seed: int, num_features: Num,):
        super().__init__(seed)

        self.num_features = num_features  # this could be int or float

        # Implementor is decided when data becomes available in fit()
        self.imp = None

    def get_model_args(self, selection_method) -> Tuple:

        # Pack model argument
        return selection_method.num_features

    def dispatch_model(self, labels: pd.Series, *args):

        # Unpack model argument
        num_features = args[0]
        self.num_features = num_features
        

    def fit(self, data: pd.DataFrame, labels: pd.Series) -> NoReturn:

        check_true(len(np.unique(labels)) == 2, TypeError("Only binary labels are supported for KL Divergence"))
        
        kl_mat = np.zeros((data.shape[1], 1))
        data = data.values
        label_categories = np.unique(labels)
        
        for i in tqdm(range(data.shape[1])):
            
            pos_idx = np.where(labels == label_categories[0])[0]
            neg_idx = np.where(labels == label_categories[1])[0]
            
            f1 = np.histogram(data[pos_idx, i], bins = 100)[0]
            f2 = np.histogram(data[neg_idx, i], bins = 100)[0]
        
            f1 = f1 / np.sum(f1)
            f2 = f2 / np.sum(f2)
        
            kl = rel_entr(f1, f2)
            kl[kl == np.inf] = 0
        
            kl_mat[i] = np.sum(kl)

        self.abs_scores = kl_mat.flatten()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        # Select top-k from data based on abs_scores and num_features
        return self.get_top_k(data, self.abs_scores)
