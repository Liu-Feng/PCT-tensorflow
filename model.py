# -*-coding:utf-8-*-
# @Time    : 2021/1/20 下午7:44
# @Author  : LiuFeng
# @File    : model.py

import tensorflow as tf
import utils.tf_util as tf_util
import utils.pointnet_util as pointnet_util


def get_placeholder(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def pct_model(point_cloud, is_training, bn_decay=None):
    """

    :param point_cloud:
    :param is_training:
    :param bn_decay:
    :return:
    """
    # point_cloud -> [batch_size, num_point, 3]
    batch_size = point_cloud.get_shape()[0].value
    point_dim = point_cloud.get_shape()[2].value

    """Input Embedding module"""
    # [batch_size, num_point, 64]
    x = tf_util.conv1d(point_cloud, 64, kernel_size=1,
                       padding='VALID', stride=1, bn=True,
                       is_biases=False, is_training=is_training,
                       scope='conv0', bn_decay=bn_decay)

    x = tf_util.conv1d(x, 64, kernel_size=1,
                       padding='VALID', stride=1, bn=True,
                       is_biases=False, is_training=is_training,
                       scope='conv1', bn_decay=bn_decay)

    """ Sample and Group """

    new_xyz, new_feature, _, _ = pointnet_util.sample_and_group(
        npoint=512, radius=0.15,
        nsample=32, xyz=point_cloud,
        points=x, knn=True, use_xyz=True)
    # print(new_xyz.shape)
    # print("new_feature.shape", new_feature.shape)

    feature_0 = local_op(new_feature, out_dim=128, scope="SG1",
                         bn_decay=bn_decay, is_training=is_training)

    new_xyz, new_feature, _, _ = pointnet_util.sample_and_group(
        npoint=256, radius=0.2,
        nsample=32, xyz=new_xyz,
        points=feature_0,
        knn=True, use_xyz=True)
    # NHC
    feature_1 = local_op(new_feature, out_dim=256, scope="SG2",
                         bn_decay=bn_decay, is_training=is_training)
    #
    # NHC
    x = pt_last(feature_1, scope="pct_layer", out_dim=256,
                bn_decay=bn_decay, is_training=is_training)

    # concat in C (NHC) axis
    x = tf.concat([x, feature_1], axis=-1)
    x = tf_util.conv1d(x, 1024, kernel_size=1,
                       padding='VALID', stride=1, bn=True,
                       is_biases=False, is_training=is_training,
                       scope='conv2', bn_decay=bn_decay, activation_fn=None)

    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = tf.reduce_max(x, axis=1)

    """
    ++++++++++++++++++++++++++++++++++++++++
                    Decoder
    ++++++++++++++++++++++++++++++++++++++++
    """
    x = tf_util.fully_connected(x, 512, bn=True, is_training=is_training, is_biases=False,
                                scope='fc1', bn_decay=bn_decay)
    x = tf_util.dropout(x, keep_prob=0.5, is_training=is_training, scope='dp1')

    x = tf_util.fully_connected(x, 256, bn=True, is_training=is_training,
                                scope='fc2', bn_decay=bn_decay)
    x = tf_util.dropout(x, keep_prob=0.5, is_training=is_training, scope='dp2')

    x = tf_util.fully_connected(x, 40, activation_fn=None, scope='fc3')
    return x


def pt_last(inputs, scope, out_dim, bn_decay, is_training):
    """

    :param inputs: NHC
    :param scope:
    :param out_dim:
    :param bn_decay:
    :param is_training:
    :return:
    """

    x = tf_util.conv1d(inputs, out_dim, kernel_size=1,
                       padding='VALID', stride=1, bn=True,
                       is_biases=False, is_training=is_training,
                       scope=scope + 'conv0', bn_decay=bn_decay)
    # NHC (32, 256, 256)
    x = tf_util.conv1d(x, out_dim, kernel_size=1,
                       padding='VALID', stride=1, bn=True,
                       is_biases=False, is_training=is_training,
                       scope=scope + 'conv1', bn_decay=bn_decay)

    x1 = sa(x, out_dim, scope + "1", bn_decay, is_training)
    x2 = sa(x1, out_dim, scope + "2", bn_decay, is_training)
    x3 = sa(x2, out_dim, scope + "3", bn_decay, is_training)
    x4 = sa(x3, out_dim, scope + "4", bn_decay, is_training)
    # concat in C (NHC) axis
    x = tf.concat([x1, x2, x3, x4], axis=-1)

    return x


def sa(inputs, out_dim, scope, bn_decay, is_training):
    # NHC
    x_q = tf_util.conv1d(inputs, out_dim // 4, kernel_size=1,
                         padding='VALID', stride=1, bn=False,
                         is_biases=False, is_training=is_training,
                         activation_fn=None, scope=scope + 'q')

    # NHC
    x_k = tf_util.conv1d(inputs, out_dim // 4, kernel_size=1,
                         padding='VALID', stride=1, bn=False,
                         is_biases=False, is_training=is_training,
                         activation_fn=None, scope=scope + 'k')
    # NCH
    x_k = tf.transpose(x_k, [0, 2, 1])

    # NHC
    x_v = tf_util.conv1d(inputs, out_dim, kernel_size=1,
                         padding='VALID', stride=1,
                         bn=False, is_training=is_training,
                         activation_fn=None, scope=scope + 'v')
    # NCH
    x_v = tf.transpose(x_v, [0, 2, 1])

    energy = tf.matmul(x_q, x_k)

    attention = tf.nn.softmax(energy, axis=- 1)
    attention = attention / (1e-9 + tf.reduce_sum(attention, axis=1, keepdims=True))
    # NCH
    x_r = tf.matmul(x_v, attention)
    # NHC
    x_r = tf.transpose(x_r, [0, 2, 1])

    x_r = tf_util.conv1d(inputs - x_r, out_dim, kernel_size=1,
                         padding='VALID', stride=1,
                         bn=True, is_training=is_training,
                         scope=scope + 'attention', bn_decay=bn_decay)
    x = inputs + x_r

    return x


def local_op(input, out_dim, scope, bn_decay, is_training):
    """

    :param input: batch_size, num_point, num_sample, num_dim
    :param out_dim:
    :return:
    """

    x = tf_util.conv2d(input, out_dim, [1, 1],
                       padding='VALID', stride=[1, 1], bn=True,
                       is_biases=False, is_training=is_training,
                       scope=scope + 'conv0', bn_decay=bn_decay)
    x = tf_util.conv2d(x, out_dim, [1, 1],
                       padding='VALID', stride=[1, 1], bn=True,
                       is_biases=False, is_training=is_training,
                       scope=scope + 'conv1', bn_decay=bn_decay)

    # replace the tf_util.max_pool2d()
    x = tf.reduce_max(x, axis=2, keepdims=False)

    return x


def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = pct_model(inputs, tf.constant(True))
        print(outputs)
