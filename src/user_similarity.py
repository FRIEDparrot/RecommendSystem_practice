import copy
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from numpy.linalg import svd   # for decomposition of new u-i matrix
from scipy.spatial import distance
import faiss
import scipy.sparse as sp
import scipy.sparse.linalg
from sklearn.metrics.pairwise import cosine_similarity
from show_msg import show_cluster_counters

class UserSimilarity():
    def __init__(self, n_clusters = 20, features = 50, batch_size = 1000, mute_output = False):
        """
        :param n_clusters: resulted cluster number to get the user category
        :param batch_size: kmeans batch size, 50 - 300 is reasonable for most cases
        :param features: the dimension of the user preference matrix after PCA decomposition
                set it according to the size of your user-item data
        :param mute_output:
        """
        self.cluster_number = n_clusters
        self.kmeans_batch_size = batch_size
        self.features = features
        self.ouput_info = not mute_output # whether to output the information
        self.model = None
        self.usr_dict = None   # user mapping infomation
        self.pref_mat = None   # user preference matrix
        self.item_max = 0

    def __get_usr_mat(self, visitor_data, item_data, item_max = None):
        """
        create the user-item matrix by visitor data and item data
        :param visitor_data:
        :param item_data:
        :param item_max:
        :return:
        """
        if np.ndim(visitor_data) != 1 or np.ndim(item_data) != 1:
            raise Exception("Input is not valid, visitor_data and item_data should be 1D vector")
        if len(visitor_data) != len(item_data):
            raise Exception("Dimension of the input data is not match : visitor_data: {}, item_data: {}".format(np.shape(visitor_data[0]), np.shape(item_data[0])))
        if item_max is None:
            item_max = max(item_data)

        usrs = visitor_data.unique()
        usr_dict = {v: i for (i, v) in enumerate(usrs)}  # the map of the user id to the index

        data = pd.DataFrame({
            "visitorid" :  visitor_data,
            "itemid" : item_data,
        })
        grouped_nums = data.groupby(['visitorid', 'itemid']).size() # use groupby to get the operation number of each user

        # calculate the index
        row_idx = [usr_dict[v] for (v, i) in grouped_nums.index]
        col_idx = [i for (v, i) in grouped_nums.index]
        nums_dict = grouped_nums.to_dict()  # change the data to dict
        data_values = np.array(list(nums_dict.values()))  # change its values to array (must use list first)

        # build the user matrix -> sparse matrix
        usr_mat = coo_matrix(
            (data_values, (row_idx, col_idx)),
            shape=(len(usrs), item_max + 1),
            dtype=np.float32
        )  # use float datatype for svd decomposition
        return usr_mat, usr_dict

    def __get_pref_mat(self, usr_mat):
        """
        get the new preference matrix (new User-Item matrix) based on sparse PCA and Minibatch KMeans clustering
        this must be called in train process
        :param usr_mat: usr preference mat (storge the user and item which has operated)
        :return:
            model:  the model of the sparse PCA and KMeans (transformed data should use this model)
            usr_pref_mat: user preference matrix
            usr_labels: the cluster labels of each unique user line
            note : usr_labels return the cluster label of each row,  but not each item
        """
        model = TruncatedSVD(n_components=self.features, algorithm='randomized')
        pca_mat_train = model.fit_transform(usr_mat)

        # MiniBatchKMeans with precomputed cosine distance matrix
        mb_clusters = MiniBatchKMeans(n_clusters=self.cluster_number, batch_size=self.kmeans_batch_size,
                                      random_state=42)
        mb_clusters.fit(pca_mat_train)  # we finally use the cosine distance for fit the model
        usr_pref_mat = mb_clusters.cluster_centers_  # pref_mat is the cluster center
        usr_labels = mb_clusters.labels_
        if self.ouput_info:
            """ print detailed infomation of the clustering model """
            print("----- usr similarity model infomation ------")
            show_cluster_counters(usr_labels, show_detailed=False)
            print("cluster inertia :", mb_clusters.inertia_)
        return model, usr_pref_mat, usr_labels

    def fit(self,
            visitor_data,
            item_data,
            item_max=None):
        """
        based on the UserCF Algorithm, build the user similarity matrix
        :param visitor_data: the visitor data, 1D vector
        :param item_data: the item data, 1D vector
        :param item_max: the number of the item, if None, it will be calculated by the unique item id
                note : if the train set and test set are not from the same item space,
                       the item_num should be the union of the item space to avoid overflow
        note : fit is get the model and user similarity matrix, but not return clustered label
        """
        self.item_max = item_max
        usr_mat, self.usr_dict = self.__get_usr_mat(visitor_data, item_data, item_max=item_max)
        self.model, self.pref_mat, self.labels_ = self.__get_pref_mat(usr_mat=usr_mat)

    def fit_transform(self,
                      visitor_data,
                      item_data,
                      item_max=None):
        """
        based on the UserCF Algorithm, build the user similarity matrix and return the cluster labels
        :param visitor_data: the visitor vector, record visitor id of the action
        :param item_data: the item vector, record the item id of the action
        :param item_max: the maximum id of the item, default is calculated by item_data
                note : if the train set and test set are not from the same item space,
                       the item_num should be the union of the item space to avoid overflow
        :return: cluster labels of each visitor
        """
        self.item_max = item_max
        usr_mat, self.usr_dict = self.__get_usr_mat(visitor_data, item_data, item_max=item_max)
        self.model, self.pref_mat, labels = self.__get_pref_mat(usr_mat=usr_mat)     # get the preference matrix

        # change labels from every row  to every entry of item_data here
        usr_idx_list = np.array([self.usr_dict[id] for id in visitor_data])
        labels_result = labels[usr_idx_list]      # the labels of the visitor
        return  np.array(labels_result)

    def transform(self,
                  visitor_data,
                  item_data,
                  metric = 'euclidean',   # eculidian is better than cosine since PCA has performed before
                  use_ANN = True,
                  ):
        """
        return the clustered User labels data of each visitor input
        :param visitor_data: visitor vector (may include new visitor)
        :param item_data:    interacted item vector (new item shouldn't exceed item_max)
        :param metric:       the metric to calculate the similarity, "euclidean"  or "cosine"
        :param use_ANN:      use Approximate Nearest Neighbor Algorithm, see ref$<self.__nearest_index>
        :return: cluster labels of each visitor
        """
        if (self.model is None):
            raise ReferenceError("The User Matrix Model is not fitted yet")
        if  max(item_data) > self.item_max:
            raise ValueError(
                "The item id is out of the range of the item_max,\n"
                "   set item_max param to the max item-id in your whole dataset."
            )
        # concate it into one matrix
        usr_mat, usr_dict = self.__get_usr_mat(visitor_data, item_data, item_max=self.item_max)
        pref_mat = self.model.transform(usr_mat)  # transform it to new ui matrix
        labels = self.__nearest_index(pref_mat, self.pref_mat, metric=metric, use_ANN=use_ANN)

        usr_idx_list = np.array([usr_dict[idx] for idx in visitor_data])  # convert from usrid to idx
        label_result = labels[usr_idx_list]      # the labels of the visitor
        return label_result

    def __nearest_index(self, mat_from, mat_target, metric = 'euclidean', use_ANN=True):
        """
        We use the nearlest Neibour to calculate the nearest index of the matrix,
        note: the input are all dense matrix, but the second dim shouldn't too large
        this function only used in transform process
        :param mat_from: the matrix to be compared
        :param mat_target: the matrix to be compared
        :param metric: the metric to be used, default is cosine
        :param use_ANN: use Approximate Nearest Neighbor Algorithm,
               much faster when the matrix to calculate nearlest is large
        :return: min_dist_idx: the nearest cluster label of each row (not each entry!!!)
        """
        if not use_ANN:
            # calculate the cosine similarity for each user in the new data
            dist = distance.cdist(mat_from, mat_target, metric=metric)
            min_dist_idx = np.argmin(dist, axis=1)
        else:
            # Approximate Nearest Neighbor using Faiss
            if metric == 'cosine':
                # Normalize the vectors in row direction for cosine similarity
                mat1_normalized = mat_from / np.linalg.norm(mat_from, axis=1, keepdims=True)
                mat2_normalized = mat_target / np.linalg.norm(mat_target, axis=1, keepdims=True)
                index = faiss.IndexFlatIP(mat_target.shape[1])  # Inner Product by line direction
            elif metric == 'euclidean':
                mat1_normalized = mat_from
                mat2_normalized = mat_target
                index = faiss.IndexFlatL2(mat_target.shape[1])  # L2 distance
            else:
                raise ValueError("Currently, only 'cosine' and 'euclidean' metric is supported for ANN.")
            # Add reference data to the index
            index.add(mat2_normalized.astype(np.float32))

            # Perform nearest neighbor search
            _, min_dist_idx = index.search(mat1_normalized.astype(np.float32), 1)
            min_dist_idx = min_dist_idx.flatten()
        return min_dist_idx
