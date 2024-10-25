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
    
    def computer(self,diff_model, user_reverse_model, item_reverse_model,  user, pos):
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
        
        # add noise to user and item
        noise_user_emb, noise_item_emb, ts, pt = self.apply_noise(ori_user_emb, ori_item_emb, diff_model)
        # reverse
        user_model_output = user_reverse_model(noise_user_emb, ori_item_emb, ts)
        item_model_output = item_reverse_model(noise_item_emb, ori_user_emb, ts)

        # get recons loss就是mse，diffusion的实现方法也是一样的，pt没啥用
        user_recons = diff_model.get_reconstruct_loss(ori_user_emb, user_model_output, pt)
        item_recons = diff_model.get_reconstruct_loss(ori_item_emb, item_model_output, pt)
        recons_loss = (user_recons + item_recons) / 2

        # update the batch user and item emb
        return user_model_output, item_model_output,recons_loss, items
    
    def computer_infer(self, user, allPos,diff_model, user_reverse_model, item_reverse_model):
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
        all_aver_item_emb = []
        for pos_item in allPos:
            item_emb = items[pos_item]
            aver_item_emb = torch.mean(item_emb, dim=0)
            all_aver_item_emb.append(aver_item_emb)
        all_aver_item_emb = torch.stack(all_aver_item_emb).to(users.device)

        noise_user_emb = user_emb

        # get generated item          
        # reverse
        noise_emb = self.apply_T_noise(all_aver_item_emb, diff_model)
        indices = list(range(self.config['sampling_steps']))[::-1]
        for i in indices:
            t = torch.tensor([i] * noise_emb.shape[0]).to(noise_emb.device)
            out = diff_model.p_mean_variance(item_reverse_model, noise_emb, noise_user_emb, t)
            if self.config['sampling_noise']:
                noise = torch.randn_like(noise_emb)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(noise_emb.shape) - 1)))
                )  # no noise when t == 0
                noise_emb = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                noise_emb = out["mean"]

        return noise_emb, items
    
    def getUsersRating(self, users, train_dict, user_reverse_model, item_reverse_model, diff_model):
        item_emb, all_items = self.computer_infer(users, train_dict, diff_model, user_reverse_model, item_reverse_model)

        item_emb = self.manifold.proj(self.manifold.expmap0(item_emb, c=self.c), c=self.c)
        all_items = self.manifold.proj(self.manifold.expmap0(all_items, c=self.c), c=self.c)
        
        item_emb1 = item_emb.detach().cpu().numpy()
        np.save('item_emb.npy', item_emb1)
        all_items1 = all_items.detach().cpu().numpy()
        np.save('all_items.npy', all_items1)

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
    
    
    def getEmbedding(self, users, pos_items, neg_items, user_reverse_model, item_reverse_model, diff_model):
        users_emb, pos_emb,recons_loss, all_items = self.computer(diff_model, user_reverse_model, item_reverse_model, users, pos_items)

        users_emb = self.manifold.proj(self.manifold.expmap0(users_emb, c=self.c), c=self.c)
        pos_emb = self.manifold.proj(self.manifold.expmap0(pos_emb, c=self.c), c=self.c)
        all_items = self.manifold.proj(self.manifold.expmap0(all_items, c=self.c), c=self.c)

        neg_emb = all_items[neg_items]

        return users_emb, pos_emb, neg_emb,recons_loss
    
    def bpr_loss(self, users, pos, neg, user_reverse_model, item_reverse_model, diff_model):
        users_emb, pos_emb, neg_emb,reconstruct_loss = self.getEmbedding(users.long(), pos.long(), neg.long(), user_reverse_model, item_reverse_model, diff_model)



        pos_scores = self.manifold.sqdist(users_emb, pos_emb, self.c)
        neg_scores = self.manifold.sqdist(users_emb, neg_emb, self.c)
        
        scores=self.manifold.fermi_dirac_decoder(pos_scores, 1, 0)
        # print("scores:",scores)
        # print("pos_scores",pos_scores)
        
        loss = pos_scores - neg_scores+ 0.2
        loss[loss < 0] = 0
        # loss = torch.sum(loss)
        return loss,reconstruct_loss,scores
    
    def apply_T_noise(self, cat_emb, diff_model):
        t = torch.tensor([self.config['steps'] - 1] * cat_emb.shape[0]).to(cat_emb.device)
        noise = torch.randn_like(cat_emb)
        noise_emb = diff_model.q_sample(cat_emb, t, noise)
        return noise_emb
    
    def apply_noise(self, user_emb, item_emb, diff_model):
        # cat_emb shape: (batch_size*3, emb_size)
        emb_size = user_emb.shape[0]
        ts, pt = diff_model.sample_timesteps(emb_size, 'uniform')
        # ts_ = torch.tensor([self.config['steps'] - 1] * cat_emb.shape[0]).to(cat_emb.device)

        # add noise to users
        user_noise = torch.randn_like(user_emb)
        item_noise = torch.randn_like(item_emb)
        user_noise_emb = diff_model.q_sample(user_emb, ts, user_noise)
        item_noise_emb = diff_model.q_sample(item_emb, ts, item_noise)
        return user_noise_emb, item_noise_emb, ts, pt

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
    
    
class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0]*2 + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, noise_emb, con_emb, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(noise_emb.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            noise_emb = F.normalize(noise_emb)
        noise_emb = self.drop(noise_emb)

        all_emb = torch.cat([noise_emb, emb, con_emb], dim=-1)

        for i, layer in enumerate(self.in_layers):
            all_emb = layer(all_emb)
            if world.config['act'] == 'tanh':
                all_emb = torch.tanh(all_emb)
            elif world.config['act'] == 'sigmoid':
                all_emb = torch.sigmoid(all_emb)
            elif world.config['act'] == 'relu':
                all_emb = F.relu(all_emb)
        for i, layer in enumerate(self.out_layers):
            all_emb = layer(all_emb)
            if i != len(self.out_layers) - 1:
                if world.config['act'] == 'tanh':
                    all_emb = torch.tanh(all_emb)
                elif world.config['act'] == 'sigmoid':
                    all_emb = torch.sigmoid(all_emb)
                elif world.config['act'] == 'relu':
                    all_emb = F.relu(all_emb)
        return all_emb
