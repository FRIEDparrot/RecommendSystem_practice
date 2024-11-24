import json
import time
import warnings
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import svds # singular value decomposition
from collections import deque
from kaggle.api import KaggleApi
import kagglehub
import pandas as pd
import os.path as  path
import treelib
import sys, os
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
import torch
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from user_similarity import UserSimilarity
from item_similarity import ItemSimilarity

"""
Retail Rocket Super Market Recommendation System 
https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
also see https://www.kaggle.com/code/aafrin/retail-rocket-recommender-system-for-beginners
"""

# set working directory  to current file
os.chdir(path.dirname(path.abspath(__file__)))

class Rocket_Market():
    def __init__(self):
        warnings.filterwarnings("ignore") # turn off warning
        # ================ parameters =================
        self.cluster_number = 1500     #  user group number (used for kmeans cluster)
        self.kmeans_batch_size = 4000  # Minibatch kmeans batchsize
        self.n_components = 50         # for TruncatedSVD (PCA components)
        self.max_recommend_item = 10   # maximum recommendation 10 items for each item related
        self.load_data()
        self.run_model(self.view_data)

    def load_data(self):
        # ================ load data ===================
        self.dataset_path = r'../ecommerce-dataset/versions/2'
        self.tree_data = pd.DataFrame(pd.read_csv(path.join(self.dataset_path, 'category_tree.csv')))
        self.events_data = pd.DataFrame(pd.read_csv(path.join(self.dataset_path, 'events.csv')))
        self.similarity_dictionary = {}
        self.model = None  # User Similarity model
        # add a column and record the timestamp
        self.events_data['actual_time'] = pd.to_datetime(self.events_data['timestamp'], unit='ms')
        self.unique_visitors = np.array(self.events_data['visitorid'].unique())
        self.unique_items = np.array(self.events_data['itemid'].unique())
        self.view_data = self.events_data.loc[self.events_data.event == "view"]
        self.cart_data = self.events_data.loc[self.events_data.event == "addtocart"]
        self.trans_data = self.events_data.loc[self.events_data.transactionid.notna()]
        self.itemid_max = max(self.events_data.itemid)
        self.visitorid_max = max(self.events_data.visitorid)

    def run_model(self, data):
        """
        train and test the recommendation model
        :return:
        """
        # ============= some other evaluations of the data =============
        tree = self.__get_category_tree()
        self.__show_most_popular_items(operation='view', topk=12, palette="pastel")
        self.__show_most_popular_items(operation='addtocart', topk=12, palette="muted")
        self.__show_most_popular_items(operation='transaction', topk=12, palette="rocket")
        # =============  train and test the recommendation model =============
        self.train_recommend_model(data, train_size = 1)  # train recommend model  with view data
        self.test_recommend_model(data, test_usr_size=0.9, test_itm_size=0.3, k_filter=3)  # test the recommend model
        self.plot_predict_results()  # plot the predict results

    def train_recommend_model(self, data, train_size = 1.0):
        """
        according to the view data, build the itemCF matrix by group
        :return:
        """
        print("======== building User-Item matrix model... =========")
        if train_size < 1:
            data_train, _ = train_test_split(data, test_size=1-train_size, random_state=42)
        else:
            data_train = data
        tm = time.time()
        self.model = UserSimilarity(
            n_clusters=self.cluster_number,
            features=self.n_components,
            batch_size=self.kmeans_batch_size,
            mute_output=False
        )
        labels = self.model.fit_transform(data_train.visitorid, data_train.itemid, item_max = self.itemid_max)

        print("---------- generating the item similarity dictionary ----------")
        self.similarity_dictionary = self.__build_itemCF_byCluster(
            np.array(data_train.visitorid),
            np.array(data_train.itemid),
            np.array(labels),
            max_recommend_item_num=self.max_recommend_item,
        )
        print("======= model build succeessfully, time cost: ", time.time() - tm, "s ======= ")

    def test_recommend_model(self, data, test_usr_size=0.85, test_itm_size=0.3, k_filter = 4):
        """
        test the recommend model by splitting the view data into train and test set
        :param data: (Pandas DataFrame) data which contains the "visitorid" and "itemid" column
               tested data would be generated in this function
        :param test_usr_size: the ratio of user of test data
        :param test_itm_size: the ratio of unappeared items generated to be tested
        :param k_filter : in the test data, user who has interaction less than k_filter items will be filtered out
        :return:
        """
        if self.model == None or (len(self.similarity_dictionary.items()) == 0):
            raise ValueError("The similarity dictionary is empty, please build the similarity dictionary first")

        # split the view data into train and test set
        _, test_usrs = train_test_split(np.unique(data.visitorid), test_size=test_usr_size, random_state=42)

        data.visitorid.isin(test_usrs)  # generate the test data

        test_data = data[data.visitorid.isin(test_usrs)].groupby("visitorid")
        print("test data shape: ", len(test_data))
        filtered_test_data = (
            # filter out item less than k_filter items
            test_data.filter(lambda x: x["itemid"].nunique() >= k_filter)
            [["visitorid", "itemid"]]  # only retain 2 columns
            .reset_index(drop=True)    # reset index
        )
        num = len(filtered_test_data)
        if (num == 0):
            raise ValueError("No test data after filter")
        print("test data size after filter : ", num)

        labels = self.model.transform(filtered_test_data.visitorid, filtered_test_data.itemid, use_ANN=True) # get the cluster labels of the test data
        cls_dict = {v: c for v,c in zip(filtered_test_data.visitorid,labels)}   # get the cluster of visitor (not repeat)

        tot_itm_num = 0
        tot_rec_itm_num = 0
        hit_itm_num = 0
        mis_itm_num = 0
        grouped_data = filtered_test_data.groupby("visitorid")
        for usr_id, usr_data in grouped_data:
            items = list(usr_data.itemid.unique())  # items of this user interacted
            sample_num = int(max(np.floor((1 - test_itm_size) * len(items)), 1))
            # generate the input and predicted labels
            label_input = random.sample(items, sample_num)       # itemid to put into the model
            label_target = list(set(items) - set(label_input))   # itemid to be predicted

            # get result from self.similarity_dictionary
            cls = cls_dict[usr_id]   # get the cluster of this user
            related_items = [
                list(val) for key,val in
                self.similarity_dictionary[cls].items()
                if key in label_input  # sample from the
            ]
            if related_items != [] :
                related_items = np.hstack(related_items)  # we flat the array by numpy.hstack
            label_pred = np.unique(related_items)

            # calculate the correct prediction numbe, use set to avoid repeat
            hit_itm_num = hit_itm_num + len(set(label_target).intersection(set(label_pred)))
            tot_itm_num = tot_itm_num + len(label_target)
            tot_rec_itm_num = tot_rec_itm_num + len(label_pred)
            mis_itm_num = mis_itm_num + len(set(label_target).difference(set(label_pred)))

        print("total item type of test", tot_itm_num)
        print("total recommended item type of test", tot_rec_itm_num)
        print("hit item type of test", hit_itm_num)
        print("miss item type of test", mis_itm_num)
        print("recommend hit percentage", hit_itm_num / tot_itm_num)
        print("recommend precision", hit_itm_num / tot_rec_itm_num)
        # storge the data
        self.result = {
            "tot_num": tot_itm_num,
            "tot_rec_num": tot_rec_itm_num,
            "hit_num": hit_itm_num,
            "mis_num": mis_itm_num,
            "hit_percent": hit_itm_num / tot_itm_num,
            "prec": hit_itm_num / tot_rec_itm_num,
        }

    def plot_predict_results(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        data1 = [self.result["hit_num"], self.result["mis_num"]]
        data2 = [self.result["hit_num"], self.result['tot_rec_num'] - self.result["hit_num"]]

        colors1 = plt.cm.viridis(np.linspace(0.35, 1, len(data1)))  # use viridis colormap
        colors2 = plt.cm.plasma(np.linspace(0.35, 1, len(data2)))

        axes[0].pie(
            data1,
            labels = ["in recommend", "not in recommend"],
            autopct=lambda pct: f"{pct:.1f}%\n{int(pct * np.sum(data1)/100)}/{np.sum(data1)}",
            explode = [0.05, 0],
            colors = colors1
        )
        axes[0].set_title("View cases from recommend", fontsize=14)
        axes[0].legend(["predicted cases", "missed cases"])

        axes[1].pie(
            data2,
            labels = ["hit cases", "not viewed cases"],
            autopct=lambda pct: f"{pct:.1f}%\n{int(pct * np.sum(data2)/100)}/{np.sum(data2)}",
            explode=[0.05, 0],
            colors = colors2
        )
        axes[1].set_title("Recommend Accuracy Information", fontsize=14)
        axes[1].legend(["cases hit view", " cases not hit"])
        plt.tight_layout()
        plt.show()

    def __build_itemCF_byCluster(self,
                                 visitor_data:np.ndarray,
                                 item_data:np.ndarray,
                                 cluster_data:np.ndarray,
                                 max_recommend_item_num=6):
        """
        get the itemCF matrix according to the cluster labels
        The cluster labels are generated by the user similarity model (UserCF model)
        :param visitor_data: the visitor data
        :param item_data: the item data
        :param cluster_data: cluster labels of each visitor, must be 1-dim array with
        note : the length of visitor_data, item_data, cluster_data must be the same
        :return: most similar item dict of each user cluster label,
                 key is the cluster label
                 value is a dict, key is the item id, value is most similiar item id
                 (if no similar item found, the item id would not appear in this dict)
        """
        if (np.ndim(visitor_data) != 1) or (np.ndim(item_data) != 1) or np.ndim(cluster_data) != 1:
            raise ValueError("The input data should be 1-dim array")
        if len(visitor_data) != len(item_data) or len(visitor_data) != len(cluster_data):
            raise ValueError("The input data should have the same length")

        num = np.max(cluster_data)
        recommend_dict = {}
        unique_labels = np.sort(np.unique(cluster_data))

        similarity_dictionary = {}
        for label in unique_labels:
            # get the data in the same user cluster
            idx = np.where(cluster_data == label)[0]
            usrid_group = visitor_data[idx]
            itmid_group = item_data[idx]

            it = ItemSimilarity()
            sim_dict = it.fit_transform(usrid_group, itmid_group, k = max_recommend_item_num)
            similarity_dictionary[label] = sim_dict
        return similarity_dictionary

    def __show_most_popular_items(self, operation, topk = 10, palette='rocket', sorted=True):
        """
        show the most viewed items
        :param topk:
        :param operation: 'view' 'addtocard' 'transaction'
        :return:
        """
        item_view_data = self.events_data.loc[self.events_data.event == operation].groupby('itemid').size().sort_values(ascending=False).iloc[0:topk]
        item_view_data = pd.DataFrame(item_view_data).reset_index()  #  reset  index is used  to change the index column to common column
        item_view_data.columns = ["itemid", "number"]

        # change the itemid to sorted categorical type
        item_view_data["itemid"] = pd.Categorical(
            item_view_data["itemid"],
            categories=item_view_data["itemid"],
            ordered=True
        )
        plt.figure(figsize=(10, 4))
        sns.barplot(x="itemid", y="number", data=item_view_data, palette=palette, hue="itemid")
        plt.legend(title="itemid", ncol= max(1, topk//6))
        plt.title(f"top {topk} items of {operation}", fontsize=14)
        plt.show()

    def __get_category_tree(self) -> treelib.Tree:
        """
        used for build the category tree for this datasheet
        """
        self.category_tree = treelib.Tree()
        self.category_tree.create_node('root', identifier='root', parent=None)
        parent_nodes = self.tree_data.iloc[np.where(pd.isna(self.tree_data['parentid']))] # the return type is a tuple
        que = deque()
        for index, node in parent_nodes.iterrows():
            cid = np.int32(node['categoryid'])
            que.append(cid)
            self.category_tree.create_node(
                identifier=str(cid),
                parent='root',
            )

        while len(que) > 0:
            cid = que.popleft()
            nodes = self.tree_data.iloc[np.where(self.tree_data['parentid'] == cid)]
            if len(nodes) == 0:
                continue
            for index, node in nodes.iterrows():
                cid = np.int32(node['categoryid'])
                pid = np.int32(node['parentid'])
                que.append(cid)
                self.category_tree.create_node(
                    identifier=str(cid),
                    parent=str(pid),
                )
        self.category_tree.show(line_type="ascii-em")
        return self.category_tree

if __name__ == "__main__":
    Rocket_Market()