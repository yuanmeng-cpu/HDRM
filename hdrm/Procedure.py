import world
import numpy as np
import torch
import utils
import random
import math
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import pdb

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = []
    recall = []
    NDCG = []
    MRR = []
    
    predicted_indices_np = np.array([p.numpy() for p in predictedIndices])
    
    for n_idx, k in enumerate(topN):
        topk_indices = predicted_indices_np[:, :k]
        precision_per_user = np.zeros(len(GroundTruth))
        recall_per_user = np.zeros(len(GroundTruth))
        ndcg_per_user = np.zeros(len(GroundTruth))
        mrr_per_user = np.zeros(len(GroundTruth))
        valid_users = 0
        for u_idx, ground_truth in enumerate(GroundTruth):
            if len(ground_truth) > 0:
                valid_users += 1
                user_topk = topk_indices[u_idx]
                hits = np.isin(user_topk, ground_truth)
                num_hits = np.sum(hits)
                precision_per_user[u_idx] = num_hits / k
                recall_per_user[u_idx] = num_hits / len(ground_truth)
                idcg_weights = 1.0 / np.log2(np.arange(2, min(len(ground_truth), k) + 2))
                idcg = np.sum(idcg_weights)
                hit_positions = np.where(hits)[0]
                if len(hit_positions) > 0:
                    dcg_weights = 1.0 / np.log2(hit_positions + 2)
                    dcg = np.sum(dcg_weights)
                    ndcg_per_user[u_idx] = dcg / idcg if idcg > 0 else 0
                first_hit = np.argmax(hits) if num_hits > 0 else -1
                if first_hit != -1:
                    mrr_per_user[u_idx] = 1.0 / (first_hit + 1)
        if valid_users > 0:
            precision.append(round(np.sum(precision_per_user) / valid_users, 4))
            recall.append(round(np.sum(recall_per_user) / valid_users, 4))
            NDCG.append(round(np.sum(ndcg_per_user) / valid_users, 4))
            MRR.append(round(np.sum(mrr_per_user) / valid_users, 4))
        else:
            precision.append(0)
            recall.append(0)
            NDCG.append(0)
            MRR.append(0)
    
    return precision, recall, NDCG, MRR

def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))

def shuffle_and_get_half_with_seed(my_list, seed_value):
    shuffled_list = my_list.copy()
    random.seed(seed_value)
    random.shuffle(shuffled_list)
    half_length = len(shuffled_list) // 2
    first_half = shuffled_list[:half_length]
    return first_half


def Test(dataset, Recmodel,user_reverse_model, item_reverse_model, diff_model, epoch, w=None, multicore=0, unbias=None):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    Recmodel: model.LightGCN
    Recmodel = Recmodel.eval()
    user_reverse_model = user_reverse_model.eval()
    item_reverse_model = item_reverse_model.eval()
    validDict = dataset.valid_dict
    testDict = dataset.test_dict
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    with torch.no_grad():
        users = list(validDict.keys())
        users_list = []
        test_rating_list = []
        valid_rating_list = []
        test_groundTrue_list = []
        valid_groundTrue_list = []

        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            test_groundTrue = [testDict[u] for u in batch_users]
            valid_groundTrue = [validDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            valid_rating = Recmodel.getUsersRating(batch_users_gpu, allPos, user_reverse_model, item_reverse_model, diff_model)

            valid_exclude_index = []
            valid_exclude_items = []
            valid_items = dataset.getUserValidItems(batch_users) # exclude validation items
            for range_i, items in enumerate(allPos):
                valid_exclude_index.extend([range_i] * len(items))
                valid_exclude_items.extend(items)

            test_exclude_index = valid_exclude_index[:]
            test_exclude_items = valid_exclude_items[:]

            for range_i, items in enumerate(valid_items):
                test_exclude_index.extend([range_i] * len(items))
                test_exclude_items.extend(items)

            test_rating = valid_rating.clone()
            valid_rating[valid_exclude_index, valid_exclude_items] = -(1<<10)
            test_rating[test_exclude_index, test_exclude_items] = -(1<<10)

            _, test_rating_K = torch.topk(test_rating, k=max_K)
            _, valid_rating_K = torch.topk(valid_rating, k=max_K)
            test_rating = test_rating_K.cpu().numpy()
            valid_rating = valid_rating_K.cpu().numpy()

            del test_rating, valid_rating
            users_list.append(batch_users)

            test_rating_list.extend(test_rating_K.cpu())
            valid_rating_list.extend(valid_rating_K.cpu()) # shape: n_batch, user_bs, max_k
            test_groundTrue_list.extend(test_groundTrue)
            valid_groundTrue_list.extend(valid_groundTrue)
        #ipdb.set_trace()
        assert total_batch == len(users_list)
        test_precision, test_recall, test_NDCG, test_MRR = computeTopNAccuracy(test_groundTrue_list,test_rating_list,[10,20,50,100])
        valid_precision, valid_recall, valid_NDCG, valid_MRR = computeTopNAccuracy(valid_groundTrue_list, valid_rating_list, [10,20,50,100])
        if multicore == 1:
            pool.close()
        return valid_precision, valid_recall, valid_NDCG, valid_MRR, test_precision, test_recall, test_NDCG, test_MRR

def print_results_all(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))
        
def Test_all(dataset, Recmodel,user_reverse_model, item_reverse_model, diff_model, epoch, w=None, multicore=0, flag=None, unbias=None):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    user_reverse_model = user_reverse_model.eval()
    item_reverse_model = item_reverse_model.eval()
    if flag == 0:
        testDict = dataset.valid_dict
    else:
        testDict = dataset.test_dict
    if unbias == 1:
        testDict = dataset.unbias_dict
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)

    with torch.no_grad():
        users = list(testDict.keys())

        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            rating = Recmodel.getUsersRating(batch_users_gpu, allPos, user_reverse_model, item_reverse_model, diff_model)
            exclude_index = []
            exclude_items = []
            valid_items = dataset.getUserValidItems(batch_users) # exclude validation items
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            if flag:
                for range_i, items in enumerate(valid_items):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()

            del rating
            users_list.append(batch_users)
            rating_list.extend(rating_K.cpu()) # shape: n_batch, user_bs, max_k
            groundTrue_list.extend(groundTrue)
        assert total_batch == len(users_list)
        precision, recall, NDCG, MRR = computeTopNAccuracy(groundTrue_list,rating_list,[10,20,50,100])
    
        if multicore == 1:
            pool.close()
        return precision, recall, NDCG, MRR

def print_epoch_result(results):
    print("Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                    '-'.join([str(x) for x in results['precision']]), 
                                    '-'.join([str(x) for x in results['recall']]), 
                                    '-'.join([str(x) for x in results['ndcg']])))

def print_results(result):
    """output the evaluation results."""
    print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                        '-'.join([str(x) for x in result[0]]), 
                        '-'.join([str(x) for x in result[1]]), 
                        '-'.join([str(x) for x in result[2]]), 
                        '-'.join([str(x) for x in result[3]])))
    print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                        '-'.join([str(x) for x in result[4]]), 
                        '-'.join([str(x) for x in result[5]]), 
                        '-'.join([str(x) for x in result[6]]), 
                        '-'.join([str(x) for x in result[7]])))
    
     
def print_results_group(i, loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if i is not None:
        if valid_result is not None: 
            print("[Valid_group{}]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                i,
                                '-'.join([str(x) for x in valid_result[0]]), 
                                '-'.join([str(x) for x in valid_result[1]]), 
                                '-'.join([str(x) for x in valid_result[2]]), 
                                '-'.join([str(x) for x in valid_result[3]])))
        if test_result is not None: 
            print("[Test_group{}]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                i,
                                '-'.join([str(x) for x in test_result[0]]), 
                                '-'.join([str(x) for x in test_result[1]]), 
                                '-'.join([str(x) for x in test_result[2]]), 
                                '-'.join([str(x) for x in test_result[3]])))

    else:
        if valid_result is not None: 
            print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                '-'.join([str(x) for x in valid_result[0]]), 
                                '-'.join([str(x) for x in valid_result[1]]), 
                                '-'.join([str(x) for x in valid_result[2]]), 
                                '-'.join([str(x) for x in valid_result[3]])))
        if test_result is not None: 
            print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                '-'.join([str(x) for x in test_result[0]]), 
                                '-'.join([str(x) for x in test_result[1]]), 
                                '-'.join([str(x) for x in test_result[2]]), 
                                '-'.join([str(x) for x in test_result[3]])))
