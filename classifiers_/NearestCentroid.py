###############################################################################
# IMPORTS
###############################################################################
import torch
import numpy as np
import torch.nn as nn
###############################################################################
# NEAREST CENTROID CLASSIFIER
###############################################################################

class NCC():
    def __init__(self, n_way, k_shot, q_queries, dist_func='l2', device='cuda'):

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.dist_func = dist_func
        self.device = device
        self.name = 'NCC'

        self.loss_fn = nn.NLLLoss().to(self.device)

    def fixed_length(self, x, y):
        batched_supports = x[:, :self.k_shot*self.n_way]
        batched_queries = x[:, self.k_shot*self.n_way:]

        y_queries = y[:, self.k_shot*self.n_way:]

        prototypes = self.batched_compute_prototypes(support=batched_supports, 
                n=self.n_way,
                k=self.k_shot,
                batch_size=batched_supports.shape[0])

        # distances = -(self.pairwise_distances(x=x_queries,
        #     y=prototypes,
        #     matching_fn=self.dist_func))

        n_x = batched_queries.shape[1]
        n_y = prototypes.shape[1]

        l2_distances = -(
            batched_queries.unsqueeze(2).expand(batched_supports.shape[0], n_x, n_y, -1) - 
            prototypes.unsqueeze(1).expand(batched_supports.shape[0], n_x, n_y, -1)
        ).pow(2).sum(dim=3)

        #print(batched_queries.unsqueeze(2).expand(batched_supports.shape[0], n_x, n_y, -1).shape)
        y_pred = l2_distances.softmax(dim=2)

        #
        log_p_y =(l2_distances).log_softmax(dim=2).squeeze()
        loss = self.loss_fn(log_p_y, y_queries)
        #

        accs = self.batched_catagorical_accuracy(y_queries, y_pred).detach().cpu().numpy()
        return accs.mean(), loss.mean()

    def fixed_return_pred_labels(self, x, y):
        batched_supports = x[:, :self.k_shot*self.n_way]
        batched_queries = x[:, self.k_shot*self.n_way:]

        y_queries = y[:, self.k_shot*self.n_way:]

        prototypes = self.batched_compute_prototypes(support=batched_supports, 
                n=self.n_way,
                k=self.k_shot,
                batch_size=batched_supports.shape[0])

        n_x = batched_queries.shape[1]
        n_y = prototypes.shape[1]

        l2_distances = -(
            batched_queries.unsqueeze(2).expand(batched_supports.shape[0], n_x, n_y, -1) - 
            prototypes.unsqueeze(1).expand(batched_supports.shape[0], n_x, n_y, -1)
        ).pow(2).sum(dim=3)

        #print(batched_queries.unsqueeze(2).expand(batched_supports.shape[0], n_x, n_y, -1).shape)
        y_pred = l2_distances.softmax(dim=2)
        predictions = y_pred.argmax(dim=2).view(y_queries.shape)
        return predictions 

    def batched_compute_prototypes(self, support, n, k, batch_size):
        # Reshape so the first dimension indexes by class then take the mean
        # along that dimension to generate the "prototypes" for each class
        class_prototypes = support.reshape(batch_size, n, k, -1).mean(dim=2)
        return class_prototypes

    def batched_catagorical_accuracy(self, targets, predictions):
        predictions = predictions.argmax(dim=2).view(targets.shape)
        accs = torch.sum((predictions == targets), dim=1)/ targets.size(1)
        return accs



    def variable_length(self, x_support, x_query, q_num, y,):
        acc_total = 0
        loss_total = 0

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

            # Calculates prototypes of support vectors
            prototypes = self.compute_prototypes(x_task_train, self.n_way, self.k_shot)

            n_x = x_task_val.shape[0]
            n_y = prototypes.shape[0]

            logits = -(
                x_task_val.unsqueeze(1).expand(n_x, n_y, -1) - 
                prototypes.unsqueeze(0).expand(n_x, n_y, -1)
            ).pow(2).sum(dim=2)

            # We scale up the y task val to directly compare for loss but use majority for acc
            sub_q_nums_tens = torch.tensor(sub_q_num).to(x_support.device)
            scaled_up_query_y = torch.repeat_interleave(
                y_task_val, sub_q_nums_tens).to(x_support.device)
            query_loss = self.loss_fn(logits, scaled_up_query_y)

            soft_logits = logits.softmax(dim=1)
            query_pred = self.majority_vote(soft_logits, sub_q_nums_tens).to(
                x_support.device, dtype=torch.long)
            post_acc = self.vote_catagorical_acc(y_task_val, query_pred)

            loss_total += query_loss.item()
            acc_total += post_acc.item()

        avg_loss = loss_total/x_support.shape[0]
        avg_acc = acc_total/x_support.shape[0]
        return avg_acc, avg_loss

    def compute_prototypes(self, support, n, k):
        # Reshape so the first dimension indexes by class then take the mean
        # along that dimension to generate the "prototypes" for each class
        class_prototypes = support.reshape(n, k, -1).mean(dim=1)
        return class_prototypes

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

    


# ncc = NCC(5, 5, 1)
# x = torch.rand(size=(1, 30, 64))
# y = torch.tensor([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,0,1,2,3,4]).unsqueeze(0)

# start = time.time()
# ncc.fixed_length(x, y)
# end = time.time()
# print(end-start)


# start = time.time()
# ncc.leave_one_out(x, y)
# end = time.time()
# print(end-start)