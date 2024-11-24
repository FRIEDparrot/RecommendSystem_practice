import copy
import heapq
import time
import warnings
import numpy
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import coo_matrix
from sklearn.neighbors import KDTree
import faiss
import scipy.sparse as sp
import scipy.sparse.linalg
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sparse_dot_topn import awesome_cossim_topn

class ItemSimilarity():
    def __init__(self):
        self.usr_dict = {}
        self.item_dict = {}
        self.ui_mat = None   # this ui-mat is used to combine with new data to calculate the similarity
        self.visitor_data = np.array([])
        self.item_data = np.array([])

    def fit(self, visitor_data, item_data):
        """
        build ItemCF matrix, note we can't build a too large matrix
        :param visitor_data:
        :param item_data:
        :return:
        """
        usrs = np.unique(visitor_data)
        items = np.unique(item_data)
        # except using the "groupby", we can use Counter to count appear time of each elem
        self.usr_dict = {usr: idx for idx, usr in enumerate(usrs)}  # create a dict to map usr to index
        self.item_dict = {item: idx for idx, item in enumerate(items)}  # create a dict to map item to index
        self.visitor_data = visitor_data
        self.item_data = item_data

    def fit_transform(self,
        visitor_data: numpy.ndarray,
        item_data: numpy.ndarray,
        k = 6, metric = 'cosine'):
        """
        see
        :ref:`self.transform`
        :return: dict of item similarity
        """
        return self.transform(visitor_data, item_data, k=k, metric=metric)
    
    def transform(self,
                  visitor_data_new : numpy.ndarray,
                  item_data_new : numpy.ndarray,
                  k=6, metric='cosine',
                  return_self=False):
        """
        Use Cosine distance to get the item similarity
        :param visitor_data_new: visitor data (except the user that already fitted)
        :param item_data_new: item data (except the item that already fitted)
        :param k: the max number of items to be filtered  first time
        :param metric: "cosine" or "euclidean" (cosine is better than euclidean)
        :param return_self: if True, the item itself will be included in the result
        :return: dict of item similarity (key is item id, value is id array of similar items, may contain itself)
        """
        # combine the new usr and item data to the original data
        new_usrs = np.unique(visitor_data_new)
        usr_dict = copy.deepcopy(self.usr_dict)
        usr_num = len(self.usr_dict)
        usr_dict_new = {usr: idx + usr_num for idx, usr in enumerate(new_usrs) if usr not in self.usr_dict}  # user that not in original data
        usr_dict.update(usr_dict_new)   # combine with the new usr dict

        new_items = np.unique(item_data_new)
        item_dict = copy.deepcopy(self.item_dict)
        item_num = len(item_dict)
        usr_dict_new = {itm: idx + item_num for idx, itm in enumerate(new_items) if itm not in self.item_dict}
        item_dict.update(usr_dict_new)   # combine with the new usr dict

        # construct new sparse matrix
        rows = []
        cols = []
        cnts = []
        usrid = np.append(copy.deepcopy(self.visitor_data), visitor_data_new, axis=0) # combine the new usr data
        itmid = np.append(copy.deepcopy(self.item_data), item_data_new, axis=0)   # combine the new item data
        grouped_counts = Counter(zip(usrid, itmid))
        for key,val in grouped_counts.items():
            usrid, itmid = key
            cnt = val
            rows.append(usr_dict[usrid])
            cols.append(item_dict[itmid])
            cnts.append(cnt)
        # if the matrix is too small, it may not have enough similar items
        if len(item_dict) <= k:
            all_items = np.array(list(item_dict.keys()))
            similar_item_dict = {id:all_items for id in new_items}
            return similar_item_dict

        ui_mat = coo_matrix(
            (cnts, (rows, cols)),
            shape = (len(usr_dict), len(item_dict)),
            dtype=np.float32
        )

        if metric == "cosine":
            # normalize the matrix
            ui_mat_norm = ui_mat.T / sp.linalg.norm(ui_mat.T, axis=1).reshape(-1, 1)
            sim_mat = cosine_similarity(ui_mat_norm, dense_output=False)  # calcuate the new item CF matrix
            top_k_mat = awesome_cossim_topn(sim_mat, sim_mat, ntop=k, lower_bound=0.0)
        else:
            raise ValueError("metric must be 'cosine'")
        # get the top k similar items dict
        top_k_dict = pd.DataFrame({
            "x": top_k_mat.nonzero()[0],
            "y": top_k_mat.nonzero()[1]
        }).groupby("x")["y"].apply(list).to_dict()

        # re-map the index to item id
        idx_to_itmid = np.array(list(item_dict.keys()))
        mapped_top_k_dict = {}
        for key, val in top_k_dict.items():
            # change to id and nearlest item list
            itm_id = idx_to_itmid[key]
            itm_ls = [idx_to_itmid[idx] for idx in val if (return_self or idx != key)]
            # only return the non-empty nearleast items
            if (itm_ls):
                mapped_top_k_dict[itm_id] = np.array(itm_ls)
        return  mapped_top_k_dict
