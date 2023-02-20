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

        accs = []
        losses = []

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

            accs.append(task_acc)
            losses.append(query_loss)

        return accs, losses


    def variable_length(self, x_support, x_query, q_num, y):
        device = x_support.device
        # Move data to numpy for sklearn
        x_support = x_support.cpu().numpy()
        x_query = x_query.cpu().numpy()
        y = y.cpu().numpy()

        accs = []
        losses = []

        # Keep track of the last used q_num access idx
        last_query_idx = 0
        # Iterate over num tasks
        for idx in range(x_support.shape[0]):

            x_task_train = x_support[idx]

            sub_q_num = q_num[idx*(self.n_way*self.q_queries): (idx+1)*(self.n_way*self.q_queries)]
            q_num_sub_sum = sum(sub_q_num)
            x_task_val = x_query[last_query_idx: (last_query_idx + q_num_sub_sum)]
            # Update tracking index
            last_query_idx += q_num_sub_sum

            # y value access is same as a fixed length batching
            y_task_train = y[idx][:(self.n_way * self.k_shot)]
            y_task_val = y[idx][(self.n_way * self.k_shot):]

            clf = SGDClassifier(max_iter=self.adapt_steps, tol=1e-4, 
                learning_rate='adaptive', eta0=self.lr, l1_ratio=0, alpha=0,
                loss='log_loss', n_iter_no_change=2)
            clf.fit(x_task_train, y_task_train)

            # We scale up the y task val to directly compare for loss but use majority for acc
            sub_q_nums_tens = torch.tensor(sub_q_num).to(device)
            scaled_up_query_y = torch.repeat_interleave(
                torch.from_numpy(y_task_val).to(device), sub_q_nums_tens).to(device)


            query_logits = clf.predict_proba(x_task_val)
            query_logits = torch.from_numpy(query_logits).to(device)

            query_loss = self.loss_fn(query_logits, scaled_up_query_y)

            query_pred = query_logits.softmax(dim=1)
            query_pred = self.majority_vote(query_pred, sub_q_nums_tens).to(
                device, dtype=torch.long)

            task_acc = self.vote_catagorical_acc(torch.from_numpy(y_task_val).to(device),  query_pred).item()

            accs.append(task_acc)
            losses.append(query_loss)

        return accs, losses

    def majority_vote(self, soft_logits, query_nums):
        y_preds = soft_logits.argmax(dim=1)

        end_index = 0
        aggregrated_preds = torch.zeros(len(query_nums))
        for idx, num in enumerate(query_nums):
            slice = y_preds[end_index:(end_index + num)]
            value, indices = torch.mode(slice)
            aggregrated_preds[idx] = value
            end_index += slice.shape[0]
        return aggregrated_preds


    def vote_catagorical_acc(self, targets, predictions):
        return (predictions == targets).sum().float() / targets.size(0)
    
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
    