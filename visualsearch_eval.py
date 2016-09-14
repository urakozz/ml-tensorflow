from __future__ import print_function
import collections
import tensorflow as tf
import numpy as np
import zipfile
import pandas as pd
import os
import shutil
import http
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.utils import shuffle
from multiprocessing import Pool, Process
import threading
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve
from six.moves.urllib.parse import quote
from sklearn import cross_validation
from random import sample, choice

def wb(wshape=[None],bshape=[None], device='/cpu:0'):
    with tf.device(device):
        w = tf.get_variable("w", wshape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', bshape, initializer=tf.constant_initializer(0.0))
    print(w.name, w.device, w.get_shape().as_list())
    print(b.name, w.device, b.get_shape().as_list())
    return w, b

# Deep ranking
# http://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf
image_size = 128
num_channels = 3
margin = 0.1
batch_size = 16
embedding_size = 4096
l2_reg_norm = 5e-5


graph_con = tf.Graph()
with graph_con.as_default():

    # Input data.
    X_q   = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    X_pos = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    X_neg = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))

    X_eval = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))



    # Variables.
    with tf.variable_scope("convNetConvLayer1"):
        layer1_weights, layer1_biases = wb([3, 3, 3, 16], [16])
    with tf.variable_scope("convNetConvLayer2"):
        layer2_weights,layer2_biases = wb([3, 3, 16, 64], [64])
    with tf.variable_scope("convNetFCLayer3"):
        layer3_weights, layer3_biases = wb(
            [image_size // 4 * image_size // 4 * 64 , embedding_size], [embedding_size])

    def convNetModel(data, train=False):
        print("data_model", data.get_shape().as_list())
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, layer1_biases))
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #pool1 = tf.nn.dropout(pool1, 0.5)
        #pool1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        #print("hidden1", pool1.get_shape().as_list())

        conv2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, layer2_biases))
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #pool2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
        if train:
            pool2 = tf.nn.dropout(pool2, 0.5)
        #print(pool2.name, pool2.get_shape().as_list())

        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [-1, np.prod(shape[1:])])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases, name='convNetModel_out')
        print(hidden.name, hidden.get_shape().as_list())

        #         hidden = tf.matmul(hidden, layer4_weights) + layer4_biases
        return hidden



    #evaluation
    out_eval = convNetModel(X_eval, True)

def img(image_file):
    rgb = ndimage.imread(image_file).astype(float)
    rgb = (rgb - 255.0/2) / 255.0
    return rgb

pickle_file = "visualsearch_deep_ranking_embeddings.pickle"
embeddings_np = pickle.load(open(pickle_file, 'rb'))
sku_uniq = pickle.load(open("sku_uniq.pickle", 'rb'))

with tf.Session(graph = graph_con) as session:
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    init_op.run()
    # Restore variables from disk.
    saver.restore(session, "visualsearch_deep_ranking.ckpt")

    img_ = img(os.path.join("images_processed", "0040c2f8306361fabd4308ff9a01efb7"+".jpg"))

    feed_dict = {X_eval:[img_]}
    check_embeddings = session.run(out_eval, feed_dict=feed_dict)
    print("> check_embeddings.shape", check_embeddings.shape)
    print("> embeddings_np.T", embeddings_np.T.shape)
    similarity = np.dot(check_embeddings, embeddings_np.T)
    for i, sim in enumerate(similarity):
        closest = sim.argsort()[-10:]
        print("> closest i ",i," >>", closest, [sku_uniq[i] for i in closest])