import tensorflow as tf
import numpy as np
import scipy.io as scio
from utils.Net_ae import Net_ae
from utils.Net_ae_graph import Net_ae_graph
from utils.next_batch import next_batch1
import math
from sklearn.utils import shuffle
import timeit
import os

def model(X1, X2, X3, X1_graph, X2_graph, X3_graph, gt,
          dims_ae1, dims_ae2, dims_ae3, dims_ae1_graph, dims_ae2_graph, dims_ae3_graph,
          act, batch_size, epochs_pre, epochs_fine,
          lr_pre, lr_fine, lr_joint, alpha, beta):
    err_pre = list()
    err_total = list()
    # define each net architecture and variable(refer to framework-simplified)
    net_ae1 = Net_ae(1, dims_ae1, act)
    net_ae2 = Net_ae(2, dims_ae2, act)
    net_ae3 = Net_ae(3, dims_ae3, act)
    net_ae1_graph = Net_ae_graph(1, dims_ae1_graph, alpha, beta, act)
    net_ae2_graph = Net_ae_graph(2, dims_ae2_graph, alpha, beta, act)
    net_ae3_graph = Net_ae_graph(3, dims_ae3_graph, alpha, beta, act)

    x1_input = tf.placeholder(np.float64, [None, None])
    x2_input = tf.placeholder(np.float64, [None, None])
    x3_input = tf.placeholder(np.float64, [None, None])
    x1_input_graph = tf.placeholder(np.float64, [None, None])
    x2_input_graph = tf.placeholder(np.float64, [None, None])
    x3_input_graph = tf.placeholder(np.float64, [None, None])

    h_input = tf.placeholder(np.float64, [None, dims_ae1[-1]])
    h_list = tf.trainable_variables()

    # Step1 : layer-wise pretrain
    # layer-wise pretrain data-path
    for j in range(1, len(dims_ae1)):
        locals()['predata' + str(j)] = net_ae1.get_loss(x1_input, j) + net_ae2.get_loss(x2_input, j) + net_ae3.get_loss(x3_input, j)
        locals()['update_predata' + str(j)] = tf.train.AdamOptimizer(learning_rate=lr_pre).minimize(locals()['predata'+str(j)])
    # layer-wise pretrain graph-path
    for j in range(1, len(dims_ae1_graph)):
        locals()['pregraph' + str(j)] = net_ae1_graph.get_loss(x1_input_graph, j) + net_ae2_graph.get_loss(x2_input_graph, j) + net_ae3_graph.get_loss(x3_input_graph, j)
        locals()['update_pregraph' + str(j)] = tf.train.AdamOptimizer(learning_rate=lr_pre).minimize(locals()['pregraph' + str(j)])

    # Step2 : path-wise fine-tuning
    # path-wise fine-tuning data-path
    loss_finedata = net_ae1.loss_reconstruct(x1_input) + net_ae2.loss_reconstruct(x2_input) + net_ae3.loss_reconstruct(x3_input)
    update_finedata = tf.train.AdamOptimizer(lr_fine).minimize(loss_finedata)
    # path-wise fine-tuning graph-path
    loss_finegraph = net_ae1_graph.loss_reconstruct(x1_input_graph) + net_ae2_graph.loss_reconstruct(x2_input_graph) + net_ae3_graph.loss_reconstruct(x3_input_graph)
    update_finegraph = tf.train.AdamOptimizer(lr_fine).minimize(loss_finegraph)

    z_half1 = net_ae1.get_z_half(x1_input)
    z_half2 = net_ae2.get_z_half(x2_input)
    z_half3 = net_ae3.get_z_half(x3_input)

    # Step3 : joint fine-tuning
    # fea_latent 喂的数据为data-path的z_half; fea_rec 喂的数据为data-path的rec_x
    loss_jointfine = net_ae1_graph.loss_jointfine(x1_input, x1_input_graph, h_input, net_ae1) + net_ae2_graph.loss_jointfine(x2_input, x2_input_graph, h_input, net_ae2) + net_ae3_graph.loss_jointfine(x3_input, x3_input_graph, h_input, net_ae3)
    update_jointfine = tf.train.AdamOptimizer(lr_joint).minimize(loss_jointfine, var_list=h_list)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    # layer-wise pretrain
    X1, X1_graph, X2, X2_graph, X3, X3_graph, gt = shuffle(X1, X1_graph, X2, X2_graph, X3, X3_graph, gt)
    X1_pre, X1_pre_graph, X2_pre, X2_pre_graph, X3_pre, X3_pre_graph, gt_pre = X1, X1_graph, X2, X2_graph, X3, X3_graph, gt
    # layer-wise data-path
    print("------layer-wise the data-path------")
    for j in range(1, len(dims_ae1)):
        for i in range(epochs_pre):
            for batch_x1, batch_x2, batch_x3, batch_gt, batch_No in next_batch1(X1_pre, X2_pre, X3_pre, gt_pre, batch_size):
                _, val_predata = sess.run([locals()['update_predata' + str(j)], locals()['predata' + str(j)]],
                                                feed_dict={x1_input: batch_x1, x2_input: batch_x2, x3_input: batch_x3})
                err_pre.append(val_predata)
                output = "predata layer {:.0f}, epoch : {:.0f}, Batch : {:.0f}  ===> loss = {:.4f} ".format(j, (i + 1), batch_No, val_predata)
                print(output)
        # 上一层的输出作为下一层的输入
        X1_pre = sess.run(net_ae1.get_encoder(X1_pre, net_ae1.weights, j))
        X2_pre = sess.run(net_ae2.get_encoder(X2_pre, net_ae2.weights, j))
        X3_pre = sess.run(net_ae3.get_encoder(X3_pre, net_ae3.weights, j))
        
    # layer-wise graph-path
    print("------layer-wise the graph-path------")
    for j in range(1, len(dims_ae1_graph)):
        for i in range(epochs_pre):
            for batch_x1_graph, batch_x2_graph, batch_x3_graph, batch_gt, batch_No in next_batch1(X1_pre_graph, X2_pre_graph, X3_pre_graph, gt_pre, batch_size):
                _, val_pregraph = sess.run([locals()['update_pregraph' + str(j)], locals()['pregraph' + str(j)]],
                                       feed_dict={x1_input_graph: batch_x1_graph, x2_input_graph: batch_x2_graph, x3_input_graph: batch_x3_graph})
                err_pre.append(val_pregraph)
                output = "pregraph layer {:.0f}, epoch : {:.0f}, Batch : {:.0f}  ===> loss = {:.4f} ".format(j, (i + 1), batch_No, val_pregraph)
                print(output)
        # 上一层的输出作为下一层的输入
        X1_pre_graph = sess.run(net_ae1_graph.get_encoder(X1_pre_graph, net_ae1_graph.weights, j))
        X2_pre_graph = sess.run(net_ae2_graph.get_encoder(X2_pre_graph, net_ae2_graph.weights, j))
        X3_pre_graph = sess.run(net_ae3_graph.get_encoder(X3_pre_graph, net_ae3_graph.weights, j))

    # path-wise fine-tuning
    # path-wise finetune data-path
    print("------path-wise fine-tuning the data-path------")
    for i in range(epochs_fine):
        for batch_x1, batch_x2, batch_x3, batch_gt, batch_No in next_batch1(X1, X2, X3, gt, batch_size):
            _, val_finedata = sess.run([update_finedata, loss_finedata], feed_dict={x1_input: batch_x1, x2_input: batch_x2, x3_input: batch_x3})
            err_pre.append(val_finedata)
            output = "finedata, epoch : {:.0f}, Batch : {:.0f}  ===> loss = {:.4f} ".format((i + 1), batch_No,
                                                                                                     val_finedata)
            print(output)

    # path-wise finetune graph-path
    print("------path-wise fine-tuning the graph-path------")
    for i in range(epochs_fine):
        for batch_x1_graph, batch_x2_graph, batch_x3_graph, batch_gt, batch_No in next_batch1(X1_graph, X2_graph, X3_graph, gt, batch_size):
            _, val_finegraph = sess.run([update_finegraph, loss_finegraph],
                                       feed_dict={x1_input_graph: batch_x1_graph, x2_input_graph: batch_x2_graph, x3_input_graph: batch_x3_graph})
            err_pre.append(val_finegraph)
            output = "finegraph, epoch : {:.0f}, Batch : {:.0f}  ===> loss = {:.4f} ".format((i + 1), batch_No,
                                                                                            val_finegraph)
            print(output)
    # 将学习到的每个视图的表示通过均值的方式融合成公共表示H,首先用于初始化的H
    latent_z1 = sess.run(net_ae1.get_z_half(X1))
    latent_z2 = sess.run(net_ae2.get_z_half(X2))
    latent_z3 = sess.run(net_ae3.get_z_half(X3))
    H = []
    H.append(latent_z1)
    H.append(latent_z2)
    H.append(latent_z3)
    H = np.array(H)
    H = np.mean(H, axis=0)
    num_samples = X1.shape[0]
    num_batchs = math.ceil(num_samples / batch_size)
    print("------joint fine-tuning------")
    for i in range(epochs_fine):
        for num_batch_i in range(int(num_batchs)):
            start_idx, end_idx = num_batch_i * batch_size, (num_batch_i + 1) * batch_size
            end_idx = min(num_samples, end_idx)
            batch_x1 = X1[start_idx: end_idx, ...]
            batch_x2 = X2[start_idx: end_idx, ...]
            batch_x3 = X3[start_idx: end_idx, ...]
            batch_x1_graph = X1_graph[start_idx: end_idx, ...]
            batch_x2_graph = X2_graph[start_idx: end_idx, ...]
            batch_x3_graph = X3_graph[start_idx: end_idx, ...]
            batch_h = H[start_idx: end_idx, ...]

            _, val_jointfine = sess.run([update_jointfine, loss_jointfine], feed_dict={x1_input: batch_x1, x2_input: batch_x2, x3_input: batch_x3,
                                                                                       x1_input_graph: batch_x1_graph, x2_input_graph: batch_x2_graph, x3_input_graph: batch_x3_graph,
                                                                                       h_input: batch_h})
            err_pre.append(val_jointfine)
            output = "loss_jointfine, epoch : {:.0f}, Batch : {:.0f}  ===> loss = {:.4f} ".format((i + 1), (num_batch_i + 1),
                                                                                                     val_jointfine)
            print(output)

        # 求出优化后的两个特征表示的均值 更新H
        z_half1_new = sess.run(z_half1, feed_dict={x1_input: batch_x1})
        z_half2_new = sess.run(z_half2, feed_dict={x2_input: batch_x2})
        z_half3_new = sess.run(z_half3, feed_dict={x3_input: batch_x3})
        z_mean = []
        z_mean.append(z_half1_new)
        z_mean.append(z_half2_new)
        z_mean.append(z_half3_new)
        z_mean = np.array(z_mean)
        z_mean = np.mean(z_mean, axis=0)
        H[start_idx: end_idx, ...] = z_mean
    return H, gt
