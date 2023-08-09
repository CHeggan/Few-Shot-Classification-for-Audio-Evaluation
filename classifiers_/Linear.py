###############################################################################
# IMPORTS
###############################################################################
import sys
import time

import torch
import torch.nn as nn

###############################################################################
# LINEAR CLASSIFIER
###############################################################################
class Linear():
    def __init__(self, input_features, adapt_steps, lr, n_way, k_shot, q_queries, device):

        self.input_features = input_features
        self.adapt_steps = adapt_steps
        self.lr = lr
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.device = device

        self.name = 'linear'

        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def fixed_length(self, x, y):
        accs = []
        losses = []

        for batch_idx in range(x.shape[0]):
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]

            supports = x_batch[:self.k_shot*self.n_way]
            queries = x_batch[self.k_shot*self.n_way:]

            y_support = y_batch[:self.k_shot*self.n_way]
            y_queries = y_batch[self.k_shot*self.n_way:]

            # We create a new linear model and optimiser
            linear_model = nn.Linear(in_features=self.input_features, out_features=self.n_way, bias=False).to(self.device)
            optimiser = torch.optim.Adam(linear_model.parameters(), lr=self.lr)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.adapt_steps, 
            #     eta_min=0, verbose=False)
            #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.99)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.2,
                patience=5, threshold=1e-4)

            linear_model.train()
            for i in range(self.adapt_steps):
                optimiser.zero_grad()
                support_logits = linear_model(supports)
                support_loss = self.loss_fn(support_logits, y_support)
                support_loss.backward()
                optimiser.step()
                scheduler.step(support_loss)

            linear_model.eval()

            support_logits = linear_model(supports)
            support_loss = self.loss_fn(support_logits, y_support)

            query_logits = linear_model(queries)
            query_pred = query_logits.softmax(dim=1)
            task_acc = self.catagorical_accuracy(y_queries, query_pred) 

            accs.append(task_acc)
            losses.append(support_loss)

        return accs, losses

    def catagorical_accuracy(self, targets, predictions):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        accs = torch.sum((predictions == targets), dim=0)/ targets.size(0)
        return accs

    def leave_one_out(self, x, y):
        total_avg_acc = 0
        total_avg_loss = 0

        for batch_idx in range(x.shape[0]):
            running_pseudo_acc = 0
            running_pseudo_loss = 0

            x_batch = x[batch_idx]
            y_batch = y[batch_idx]

            supports = x_batch[:self.k_shot*self.n_way]
            y_support = y_batch[:self.k_shot*self.n_way]

            for sup_idx in range(supports.shape[0]):
                pseudo_query = supports[sup_idx].unsqueeze(0)
                y_pseudo_query = y_support[sup_idx].unsqueeze(0)

                pseudo_supports = torch.cat( [supports[0:sup_idx], supports[sup_idx+1:]] )
                y_pseudo_supports = torch.cat( [y_support[0:sup_idx], y_support[sup_idx+1:]] )
                
                # We create a new linear model and optimiser for our remaining supports
                linear_model = nn.Linear(in_features=self.input_features, out_features=self.n_way, bias=False).to(self.device)
                optimiser = torch.optim.Adam(linear_model.parameters(), lr=self.lr)

                linear_model.train()
                for i in range(self.adapt_steps):
                    optimiser.zero_grad()
                    support_logits = linear_model(pseudo_supports)
                    support_loss = self.loss_fn(support_logits, y_pseudo_supports)
                    support_loss.backward()
                    optimiser.step()

                linear_model.eval()
                query_logits = linear_model(pseudo_query)
                query_loss = self.loss_fn(query_logits, y_pseudo_query)
                query_pred = query_logits.softmax(dim=1)


                task_acc = self.catagorical_accuracy(y_pseudo_query, query_pred)
                running_pseudo_acc += task_acc
                running_pseudo_loss += query_loss
            
            avg_pseudo_acc = running_pseudo_acc/supports.shape[0]
            avg_pseudo_loss = running_pseudo_loss/supports.shape[0]

            total_avg_acc += avg_pseudo_acc
            total_avg_loss += avg_pseudo_loss

        final_avg_acc = total_avg_acc/x.shape[0]
        final_avg_loss = total_avg_loss/x.shape[0]
        return final_avg_acc, final_avg_loss

    