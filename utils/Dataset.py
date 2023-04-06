import h5py
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import scipy.io
from sklearn.metrics.pairwise import rbf_kernel
import os


class Dataset():
    def __init__(self, name):
        self.path = './dataset/'
        self.name = name

    def load_data(self):
        data_path = self.path + self.name + '.mat'
        if 'COIL' in self.name:
            dataset = scipy.io.loadmat(data_path)
            x1, x2, x3, y = dataset['x1'], dataset['x2'], dataset['x3'], dataset['gt']
            y = y - 1
            tmp = np.zeros(y.shape[0])
            y = np.reshape(y, np.shape(tmp))
            return x1, x2, x3, y

    def load_graph1(self, x1, x2, x3, K, gamma):
        Graph_file = 'Graph_%s_%d.mat' % (self.name, K)
        if os.path.isfile(Graph_file):  # 图已构造
            # load pre-trained graph from mat
            print('... loading %s ' % (Graph_file))
            mat_contents = scipy.io.loadmat(Graph_file)
            x1_graph = mat_contents['x1_graph']
            x2_graph = mat_contents['x2_graph']
            x3_graph = mat_contents['x3_graph']
        else:
            print('... construct %s ' % (Graph_file))
            for i in range(1, 4):
                if i == 1:
                    sim = rbf_kernel(x1, x1, gamma)
                    s_sim = np.sort(sim)
                    # calculate similarity, sim越大，离得越近，取sim最大的前K个值。
                    s_sim_k_value = s_sim[:, -K]
                    ss = s_sim_k_value.reshape(s_sim_k_value.shape[0], 1)
                    ss = np.dot(ss, np.ones((1, x1.shape[0])))

                    select_knn = (sim - ss) >= 0  #select K neighbour
                    x1_graph = sim * select_knn
                if i == 2:
                    sim = rbf_kernel(x2, x2, gamma)
                    s_sim = np.sort(sim)
                    # calculate similarity, sim越大，离得越近，取sim最大的前K个值
                    s_sim_k_value = s_sim[:, -K]
                    ss = s_sim_k_value.reshape(s_sim_k_value.shape[0], 1)
                    ss = np.dot(ss, np.ones((1, x2.shape[0])))

                    select_knn = (sim - ss) >= 0  # select K neighbour
                    x2_graph = sim * select_knn
                if i == 3:
                    sim = rbf_kernel(x3, x3, gamma)
                    s_sim = np.sort(sim)
                    # calculate similarity, sim越大，离得越近，取sim最大的前K个值
                    s_sim_k_value = s_sim[:, -K]
                    ss = s_sim_k_value.reshape(s_sim_k_value.shape[0], 1)
                    ss = np.dot(ss, np.ones((1, x3.shape[0])))

                    select_knn = (sim - ss) >= 0  # select K neighbour
                    x3_graph = sim * select_knn
            scipy.io.savemat(Graph_file, {'x1_graph': x1_graph, 'x2_graph': x2_graph, 'x3_graph': x3_graph})
        return x1_graph, x2_graph, x3_graph

    def normalize(self, x, min=0):
        if min == 0:
            scaler = MinMaxScaler([0, 1])   #数据归一化，收敛在[0,1]之间
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x
