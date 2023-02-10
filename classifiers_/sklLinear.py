###############################################################################
# IMPORTS
###############################################################################
import sys
import time

import torch
import scipy
import numpy as np
import torch.nn as nn
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

###############################################################################
# LINEAR CLASSIFIER
###############################################################################
class skLinear():
    def __init__(self, input_features, adapt_steps, n_way, k_shot, q_queries, device):

        self.input_features = input_features
        self.adapt_steps = adapt_steps
        self.lr = 0.1
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.device = device

        self.name = 'linear'

        self.loss_fn = nn.CrossEntropyLoss()

    def fixed_length(self, x, y):
        # Move data to numpy for sklearn
        x = x.cpu().numpy()
        y = y.cpu().numpy()

        total_acc = 0
        total_loss = 0

        for batch_idx in range(x.shape[0]):
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]

            supports = x_batch[:self.k_shot*self.n_way]
            queries = x_batch[self.k_shot*self.n_way:]

            y_support = y_batch[:self.k_shot*self.n_way]
            y_queries = y_batch[self.k_shot*self.n_way:]

            clf = SGDClassifier(max_iter=self.adapt_steps, tol=1e-4, 
                learning_rate='adaptive', eta0=self.lr, l1_ratio=0, alpha=0,
                loss='log_loss', n_iter_no_change=2)
            clf.fit(supports, y_support)

            query_logits = clf.predict_proba(queries)
            query_loss = self.loss_fn(torch.from_numpy(query_logits), torch.from_numpy(y_queries))

            query_pred = clf.predict(queries)
            task_acc = accuracy_score(y_queries,  query_pred)

            total_acc += task_acc
            total_loss += query_loss

        avg_acc = task_acc/x.shape[0]
        query_loss = total_loss/x.shape[0]

        return [avg_acc], query_loss

    
    def return_pred_labels(self, x, y):
        # Move data to numpy for sklearn
        x = x.cpu().numpy()
        y = y.cpu().numpy()

        total_acc = 0
        total_loss = 0

        all_query_pred = []

        for batch_idx in range(x.shape[0]):
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]

            supports = x_batch[:self.k_shot*self.n_way]
            queries = x_batch[self.k_shot*self.n_way:]

            y_support = y_batch[:self.k_shot*self.n_way]
            y_queries = y_batch[self.k_shot*self.n_way:]

            clf = SGDClassifier(max_iter=self.adapt_steps, tol=1e-4, 
                learning_rate='adaptive', eta0=self.lr, l1_ratio=0, alpha=0,
                loss='log_loss', n_iter_no_change=2)
            clf.fit(supports, y_support)

            support_logits = clf.predict_proba(supports)
            support_loss = self.loss_fn(torch.from_numpy(support_logits), torch.from_numpy(y_support))

            query_pred = clf.predict(queries)

            all_query_pred.append(query_pred)


        if x.shape[0] == 1:
            return torch.from_numpy(all_query_pred[0]).to(self.device)
        else:
            return all_query_pred
    