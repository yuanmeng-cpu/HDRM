import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from time import time
import pdb
import math

import numpy as np
import scipy.sparse as sp
import torch


class BasicDataset(Dataset):
	def __init__(self):
		print("init dataset")
	
	@property
	def n_users(self):
		raise NotImplementedError
	
	@property
	def m_items(self):
		raise NotImplementedError
	
	@property
	def trainDataSize(self):
		raise NotImplementedError
	
	@property
	def validDict(self):
		raise NotImplementedError

	@property
	def testDict(self):
		raise NotImplementedError
	
	@property
	def allPos(self):
		raise NotImplementedError
	
	def getUserItemFeedback(self, users, items):
		raise NotImplementedError
	
	def getUserPosItems(self, users):
		raise NotImplementedError
	
	def getUserNegItems(self, users):
		"""
		not necessary for large dataset
		it's stupid to return all neg items in super large dataset
		"""
		raise NotImplementedError
	
	def getSparseGraph(self):
		"""
		build a graph in torch.sparse.IntTensor.
		Details in NGCF's matrix form
		A = 
			|I,   R|
			|R^T, I|
		"""
		raise NotImplementedError

class DiffData(BasicDataset):

	def __init__(self,config = world.config, path = None):
		# train or test
		print(f'loading [{path}]')
		self.num_ng = config['num_ng']
		self.split = config['A_split']
		self.folds = config['A_n_fold']
		self.mode_dict = {'train': 0, "test": 1}
		self.mode = self.mode_dict['train']
		self.n_user = 0
		self.m_item = 0
		train_file = path + '/train_list.npy'
		valid_file = path + '/valid_list.npy'
		test_file = path + '/test_list.npy'

		self.path = path
		trainUniqueUsers, trainItem, trainUser = [], [], []
		validUniqueUsers, validItem, validUser = [], [], []
		testUniqueUsers, testItem, testUser = [], [], []
		self.traindataSize = 0
		self.validDataSize = 0
		self.testDataSize = 0

		self.train_list = np.load(train_file, allow_pickle=True)
		self.valid_list = np.load(valid_file, allow_pickle=True)
		self.test_list = np.load(test_file, allow_pickle=True)

		self.train_dict = {}
		self.valid_dict = {}
		self.test_dict = {}
		for uid, iid in self.train_list:
			if uid not in self.train_dict:
				self.train_dict[uid] = []
			self.train_dict[uid].append(iid)

		for uid, iid in self.valid_list:
			if uid not in self.valid_dict:
				self.valid_dict[uid] = []
			self.valid_dict[uid].append(iid)
		
		for uid, iid in self.test_list:
			if uid not in self.test_dict:
				self.test_dict[uid] = []
			self.test_dict[uid].append(iid)

		for uid in self.train_dict.keys():
			trainUniqueUsers.append(uid)
			trainUser.extend([uid] * len(self.train_dict[uid]))
			trainItem.extend(self.train_dict[uid])
			self.m_item = max(self.m_item, max(self.train_dict[uid]))
			self.n_user = max(self.n_user, uid)
			self.traindataSize += len(self.train_dict[uid])

		self.trainUniqueUsers = np.array(trainUniqueUsers)
		self.trainUser = np.array(trainUser)
		self.trainItem = np.array(trainItem)

		for uid in self.valid_dict.keys():
			if len(self.valid_dict[uid]) != 0:
				validUniqueUsers.append(uid)
				validUser.extend([uid] * len(self.valid_dict[uid]))
				validItem.extend(self.valid_dict[uid])
				self.m_item = max(self.m_item, max(self.valid_dict[uid]))
				self.n_user = max(self.n_user, uid)
				self.validDataSize += len(self.valid_dict[uid])
		self.validUniqueUsers = np.array(validUniqueUsers)
		self.validUser = np.array(validUser)
		self.validItem = np.array(validItem)

		for uid in self.test_dict.keys():
			if len(self.test_dict[uid]) != 0:
				testUniqueUsers.append(uid)
				testUser.extend([uid] * len(self.test_dict[uid]))
				testItem.extend(self.test_dict[uid])
				self.m_item = max(self.m_item, max(self.test_dict[uid]))
				self.n_user = max(self.n_user, uid)
				self.testDataSize += len(self.test_dict[uid])
		self.testUniqueUsers = np.array(testUniqueUsers)
		self.testUser = np.array(testUser)
		self.testItem = np.array(testItem)
		self.m_item += 1
		self.n_user += 1
		
		self.Graph = None
		print(f"{self.trainDataSize} interactions for training")
		print(f"{self.validDataSize} interactions for validation")
		print(f"{self.testDataSize} interactions for testing")
		print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.validDataSize + self.testDataSize) / self.n_users / self.m_items}")

		# (users,items), bipartite graph
		self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
									  shape=(self.n_user, self.m_item))
		self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
		self.users_D[self.users_D == 0.] = 1
		self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
		self.items_D[self.items_D == 0.] = 1.
		# pre-calculate
		self._allPos = self.getUserPosItems(list(range(self.n_user)))
		print(f"{world.dataset} is ready to go")

	@property
	def n_users(self):
		return self.n_user
	
	@property
	def m_items(self):
		return self.m_item
	
	@property
	def trainDataSize(self):
		return self.traindataSize
	

	@property
	def trainDict(self):
		return self.train_dict

	@property
	def validDict(self):
		return self.valid_dict

	@property
	def testDict(self):
		return self.test_dict

	@property
	def allPos(self):
		return self._allPos

	def _split_A_hat(self,A):
		A_fold = []
		fold_len = (self.n_users + self.m_items) // self.folds
		for i_fold in range(self.folds):
			start = i_fold*fold_len
			if i_fold == self.folds - 1:
				end = self.n_users + self.m_items
			else:
				end = (i_fold + 1) * fold_len
			A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
		return A_fold

	def _convert_sp_mat_to_sp_tensor(self, X):
		coo = X.tocoo().astype(np.float32)
		row = torch.Tensor(coo.row).long()
		col = torch.Tensor(coo.col).long()
		index = torch.stack([row, col])
		data = torch.FloatTensor(coo.data)
		return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
		
	def getSparseGraph(self):
		print("loading adjacency matrix")
		
		if self.Graph is None:
			try:
				pre_adj_mat = sp.load_npz(self.path + '/adj_csr.npz')
				print("successfully loaded...")
				adj_csr = pre_adj_mat
				self.Graph = normalize(adj_csr + sp.eye(adj_csr.shape[0]))
				self.Graph = sparse_mx_to_torch_sparse_tensor(self.Graph).to(world.device)
				print(" type Graph", type(self.Graph))
			except:
				print("generating adjacency matrix")
				s = time()
				coo_user_item = self.UserItemNet.tocoo()
				rows = np.concatenate((coo_user_item.row, coo_user_item.col + self.n_users))
				cols = np.concatenate((coo_user_item.col + self.n_users, coo_user_item.row))
				data = np.ones((coo_user_item.nnz * 2,))
				adj_csr = sp.coo_matrix((data, (rows, cols)),
									shape=(self.n_users + self.m_items, self.n_users + self.m_items)).tocsr().astype(np.float32)
				end = time()
				print(f"costing {end - s}s, saved norm_mat...")
				sp.save_npz(self.path + '/adj_csr.npz', adj_csr)
				self.Graph = normalize(adj_csr + sp.eye(adj_csr.shape[0]))
				self.Graph = sparse_mx_to_torch_sparse_tensor(self.Graph).to(world.device)
				print(" type Graph", type(self.Graph))
		return self.Graph

	def getUserItemFeedback(self, users, items):
		"""
		users:
			shape [-1]
		items:
			shape [-1]
		return:
			feedback [-1]
		"""
		# print(self.UserItemNet[users, items])
		return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

	def getUserPosItems(self, users):
		posItems = []
		for user in users:
			posItems.append(self.UserItemNet[user].nonzero()[1])
		return posItems

	def getUserValidItems(self, users):
		validItems = []
		for user in users:
			if user in self.valid_dict:
				validItems.append(self.valid_dict[user])
		return validItems
	
	def get_pair_bpr(self):
		"""
		the original impliment of BPR Sampling in LightGCN
		:return:
			np.array
		"""
		user_num = self.traindataSize
		users = np.random.randint(0, self.n_users, user_num)
		self.user = []
		self.posItem = []
		self.negItem = []
		for i, user in enumerate(users):
			posForUser = self._allPos[user]
			if len(posForUser) == 0:
				continue
			posindex = np.random.randint(0, len(posForUser))
			positem = posForUser[posindex]
			while True:
				negitem = np.random.randint(0, self.m_items)
				if negitem in posForUser:
					continue
				else:
					break
			self.user.append(user)
			self.posItem.append(positem)
			self.negItem.append(negitem)

	def __getitem__(self, idx):
		return self.user[idx], self.posItem[idx], self.negItem[idx]
	
	def __len__(self):
		return self.traindataSize

    
def normalize(mx):
    """Row-normalize sparse matrix."""
    """https://github.com/HazyResearch/hgcn/blob/a526385744da25fc880f3da346e17d0fe33817f8/utils/data_utils.py"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    """https://github.com/HazyResearch/hgcn/blob/a526385744da25fc880f3da346e17d0fe33817f8/utils/data_utils.py"""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
