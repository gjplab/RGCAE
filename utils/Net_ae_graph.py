import tensorflow as tf
from tensorflow.contrib import layers

class Net_ae_graph(object):
    def __init__(self, v, dims_encoder, alpha, beta, activation, reg=None):
        self.v = v
        self.dims_encoder = dims_encoder
        self.dims_decoder = [i for i in reversed(dims_encoder)]
        self.num_layers = len(self.dims_encoder)
        self.alpha = alpha
        self.beta = beta
        self.activation = activation
        self.reg = reg
        if activation in ['tanh', 'sigmoid']:
            self.initializer = layers.xavier_initializer()
        if activation == 'relu':
            self.initializer = layers.variance_scaling_initializer(mode='FAN_AVG')

        self.weights, self.netpara = self.init_weights()

    def init_weights(self):
        all_weights = dict()
        with tf.variable_scope("aenet"):
            for i in range(1, self.num_layers):
                all_weights['g_enc' + str(self.v) + '_w' + str(i)] = tf.get_variable("g_enc" + str(self.v) + "_w" + str(i),
                                                                                   shape=[self.dims_encoder[i - 1],
                                                                                          self.dims_encoder[i]],
                                                                                   dtype=tf.float64,
                                                                                   initializer=self.initializer,
                                                                                   regularizer=self.reg)
                all_weights['g_enc' + str(self.v) + '_b' + str(i)] = tf.Variable(
                    tf.zeros([self.dims_encoder[i]], dtype=tf.float64))

            for i in range(1, self.num_layers):
                all_weights['g_dec' + str(self.v) + '_w' + str(i)] = tf.get_variable("g_dec" + str(self.v) + "_w" + str(i),
                                                                                   shape=[self.dims_decoder[i - 1],
                                                                                          self.dims_decoder[i]],
                                                                                   dtype=tf.float64,
                                                                                   initializer=self.initializer,
                                                                                   regularizer=self.reg)
                all_weights['g_dec' + str(self.v) + '_b' + str(i)] = tf.Variable(
                    tf.zeros([self.dims_decoder[i]], dtype=tf.float64))
            aenet = tf.trainable_variables()    #返回使用trainable=True创建的所有变量.
        return all_weights, aenet

    def encoder(self, x, weights):
        layer = tf.add(tf.matmul(x, weights['g_enc' + str(self.v) + '_w1']), weights['g_enc' + str(self.v) + '_b1'])
        if self.activation == 'sigmoid':
            layer = tf.nn.sigmoid(layer)
        if self.activation == 'tanh':
            layer = tf.nn.tanh(layer)
        if self.activation == 'relu':
            layer = tf.nn.relu(layer)
        for i in range(2, self.num_layers):
            layer = tf.add(tf.matmul(layer, weights['g_enc' + str(self.v) + '_w' + str(i)]),
                           weights['g_enc' + str(self.v) + '_b' + str(i)])
            # if i < self.num_layers-1:
            if self.activation == 'sigmoid':
                layer = tf.nn.sigmoid(layer)
            if self.activation == 'tanh':
                layer = tf.nn.tanh(layer)
            if self.activation == 'relu':
                layer = tf.nn.relu(layer)
        return layer

    def decoder(self, z_half, weights):
        layer = tf.add(tf.matmul(z_half, weights['g_dec' + str(self.v) + '_w1']), weights['g_dec' + str(self.v) + '_b1'])
        if self.activation == 'sigmoid':
            layer = tf.nn.sigmoid(layer)
        if self.activation == 'tanh':
            layer = tf.nn.tanh(layer)
        if self.activation == 'relu':
            layer = tf.nn.relu(layer)
        for i in range(2, self.num_layers):
            layer = tf.add(tf.matmul(layer, weights['g_dec' + str(self.v) + '_w' + str(i)]),
                           weights['g_dec' + str(self.v) + '_b' + str(i)])
            # if i < self.num_layers-1:
            if self.activation == 'sigmoid':
                layer = tf.nn.sigmoid(layer)
            if self.activation == 'tanh':
                layer = tf.nn.tanh(layer)
            if self.activation == 'relu':
                layer = tf.nn.relu(layer)
        return layer

    def get_encoder(self, x, weights, j):
        layer = tf.add(tf.matmul(x, weights['g_enc' + str(self.v) + '_w' + str(j)]), weights['g_enc' + str(self.v) + '_b' + str(j)])
        if self.activation == 'sigmoid':
            layer = tf.nn.sigmoid(layer)
        if self.activation == 'tanh':
            layer = tf.nn.tanh(layer)
        if self.activation == 'relu':
            layer = tf.nn.relu(layer)
        return layer

    def get_decoder(self, z_half, weights, j):
        layer = tf.add(tf.matmul(z_half, weights['g_dec' + str(self.v) + '_w' + str(self.num_layers-j)]), weights['g_dec' + str(self.v) + '_b' + str(self.num_layers-j)])
        if self.activation == 'sigmoid':
            layer = tf.nn.sigmoid(layer)
        if self.activation == 'tanh':
            layer = tf.nn.tanh(layer)
        if self.activation == 'relu':
            layer = tf.nn.relu(layer)
        return layer

    def get_loss(self, x, j):
        z_half = self.get_encoder(x, self.weights, j)
        z = self.get_decoder(z_half, self.weights, j)
        # 重构输入loss_r
        loss_r = tf.reduce_mean(tf.pow(tf.subtract(x, z), 2.0))
        # loss_r = tf.reduce_mean(- tf.reduce_sum(x * tf.log(z) + (1 - x) * tf.log(1 - z), axis=1))
        return loss_r

    def loss_reconstruct(self, x):
        z_half = self.encoder(x, self.weights)
        z = self.decoder(z_half, self.weights)
        loss = tf.reduce_mean(tf.pow(tf.subtract(x, z), 2.0))
        return loss

    def get_z_half(self, x):
        return self.encoder(x, self.weights)

    def get_z(self, x):
        z_half = self.encoder(x, self.weights)
        return self.decoder(z_half, self.weights)


    def loss_jointfine(self, x, x_graph, h, net_ae):
        # 1. loss_finedata
        data_hidden = net_ae.encoder(x, net_ae.weights)
        rec_data = net_ae.decoder(data_hidden, net_ae.weights)
        loss_data = tf.reduce_mean(tf.pow(tf.subtract(x, rec_data), 2.0))
        # 2. loss_jointgraph : reconstructed graph from data code layer
        # construct the shared graph-decoder for joint training
        rec_graph = self.decoder(data_hidden, self.weights)
        loss_jointgraph = tf.reduce_mean(tf.pow(tf.subtract(x_graph, rec_graph), 2.0))
        # 3. loss_com
        loss_com = tf.reduce_mean(tf.pow(tf.subtract(data_hidden, h), 2.0))
        # loss_joint
        loss_joint = loss_data + self.alpha * loss_jointgraph + self.beta * loss_com
        return loss_joint
