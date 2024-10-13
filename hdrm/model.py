import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import pdb
import math
import time
import torch.nn.functional as F

import manifolds

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        raise NotImplementedError
    

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):

        self.c = torch.tensor(self.config['c'])
        self.manifold = getattr(manifolds, "Hyperboloid")()

        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        
        self.scale=0.1
        
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            print('use NORMAL distribution initilizer')
            
            self.embedding_user.state_dict()['weight'].uniform_(-self.scale, self.scale)
            self.embedding_user.weight = nn.Parameter(self.manifold.expmap0(self.embedding_user.state_dict()['weight'], self.c))
            self.embedding_user.weight = manifolds.ManifoldParameter(self.embedding_user.weight, True, self.manifold, self.c)

            self.embedding_item.state_dict()['weight'].uniform_(-self.scale, self.scale)
            self.embedding_item.weight = nn.Parameter(self.manifold.expmap0(self.embedding_item.state_dict()['weight'], self.c))
            self.embedding_item.weight = manifolds.ManifoldParameter(self.embedding_item.weight, True, self.manifold, self.c)

            
            
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")
    
    def computer(self, user, pos):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        users_emb_hpy = self.manifold.proj(users_emb, c=self.c)
        items_emb_hpy = self.manifold.proj(items_emb, c=self.c)

        users_emb = self.manifold.logmap0(users_emb_hpy, c=self.c)
        items_emb = self.manifold.logmap0(items_emb_hpy, c=self.c)


        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        
        g_droped = self.Graph    
        
        
        for i in range(self.n_layers):
            # print(all_emb.shape)
            embs.append(torch.spmm(g_droped, embs[i]))

        light_out = sum(embs[1:])
        
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        # get batch user and item emb
        ori_user_emb = users[user]
        ori_item_emb = items[pos]

        # update the batch user and item emb
        return ori_user_emb, ori_item_emb, items
    
    def computer_infer(self, user, allPos):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        users_emb_hpy = self.manifold.proj(users_emb, c=self.c)
        items_emb_hpy = self.manifold.proj(items_emb, c=self.c)

        users_emb = self.manifold.logmap0(users_emb_hpy, c=self.c)
        items_emb = self.manifold.logmap0(items_emb_hpy, c=self.c)

        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        g_droped = self.Graph  
          
        for i in range(self.n_layers):
            # print(all_emb.shape)
            embs.append(torch.spmm(g_droped, embs[i]))

        light_out = sum(embs[1:])
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        user_emb = users[user.long()]

        return user_emb, items
    
    def getUsersRating(self, users, train_dict):
        item_emb, all_items = self.computer_infer(users, train_dict)

        item_emb = self.manifold.proj(self.manifold.expmap0(item_emb, c=self.c), c=self.c)
        all_items = self.manifold.proj(self.manifold.expmap0(all_items, c=self.c), c=self.c)

        rating = self.rounding_inner(item_emb, all_items)
        return rating
    
    def rounding_inner(self, item_emb, all_items):
        num_users=item_emb.shape[0]
        num_items=all_items.shape[0]

        probs_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            emb_in = item_emb[i, :]
            emb_in = emb_in.repeat(num_items).view(num_items, -1)

            emb_out = all_items
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)

            #通过负号把距离转化为概率，距离越小概率越大
            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[i] = np.reshape(probs, [-1, ])

        return torch.from_numpy(probs_matrix)
        # item_emb_expanded = item_emb.unsqueeze(1)  # Shape: [bs_user, 1, emb]
        # all_items_expanded = all_items.unsqueeze(0)  # Shape: [1, item_num, emb]

        # Element-wise multiplication
        # dot_product = torch.sum(item_emb_expanded * all_items_expanded, dim=2) 

        # return dot_product
    
    
    def getEmbedding(self, users, pos_items, neg_items):
        users_emb, pos_emb, all_items = self.computer(users, pos_items)

        users_emb = self.manifold.proj(self.manifold.expmap0(users_emb, c=self.c), c=self.c)
        pos_emb = self.manifold.proj(self.manifold.expmap0(pos_emb, c=self.c), c=self.c)
        all_items = self.manifold.proj(self.manifold.expmap0(all_items, c=self.c), c=self.c)

        neg_emb = all_items[neg_items]

        return users_emb, pos_emb, neg_emb
    
    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb = self.getEmbedding(users.long(), pos.long(), neg.long())



        pos_scores = self.manifold.sqdist(users_emb, pos_emb, self.c)
        neg_scores = self.manifold.sqdist(users_emb, neg_emb, self.c)
        
        loss = pos_scores - neg_scores+ 0.2
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss