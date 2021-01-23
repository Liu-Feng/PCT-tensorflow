# -*-coding:utf-8-*-
# @Time    : 2020/10/31 下午3:32
# @Author  : LiuFeng
# @File    : train_cls_modelnet.py

"""

"""
import os
import sys
import socket
import argparse
import numpy as np
import tensorflow as tf

from models import pointnet2 as model
import utils.provider as provider

'''
some libs about modelnet dataset
'''

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 201]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 2048]')
parser.add_argument('--category', default=None, help='Which single class to train on [default: None]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--no_rotation', action='store_true', help='Disable random rotation during training.')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')

# under change
parser.add_argument('--model_file', default='pointnet2', help='Model name [default: model]')
parser.add_argument('--log_dir', default='cls_log/pointnet2_log', help='Log dir [default: log]')
parser.add_argument('--train_file', default='train', help='Model name [default: model]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BATCH_SIZE = FLAGS.batch_size
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BASE_LEARNING_RATE = FLAGS.learning_rate

pwd = os.getcwd()
BASE_DIR = pwd

LOG_DIR = FLAGS.log_dir
LOG_DIR = os.path.join(BASE_DIR, os.path.join("logs", LOG_DIR))
os.makedirs(LOG_DIR, exist_ok=True)

TRAIN_FILE = FLAGS.train_file
TRAIN_FILE = os.path.join(BASE_DIR, TRAIN_FILE + ".py")

MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model_file + '.py')

"""拷贝文件"""
os.system('cp {} {}'.format(MODEL_FILE, LOG_DIR))
os.system('cp {} {}'.format(TRAIN_FILE, LOG_DIR))

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')

LOG_FOUT.write(str(FLAGS) + '\n')

NUM_CLASSES = 40
BN_INIT_DECAY = 0.5
BN_DECAY_CLIP = 0.99
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)

HOSTNAME = socket.gethostname()

# ModelNet official train/test split
TRAIN_DATASET = provider.getDataFiles(
    os.path.join(BASE_DIR, 'datasets/modelnet40_ply_hdf5_2048/train_files.txt'))

TEST_DATASET = provider.getDataFiles(
    os.path.join(BASE_DIR, 'datasets/modelnet40_ply_hdf5_2048/test_files.txt'))


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):

            pointcloud_pl, label_pl = model.get_placeholder(BATCH_SIZE, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=[])

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar("bn_decay", bn_decay)

            pred = model.get_model(pointcloud_pl, is_training_pl, bn_decay)

            loss = model.get_loss(pred, label_pl)
            tf.summary.scalar("loss", loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.cast(label_pl, tf.int64))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32) / float(BATCH_SIZE))
            tf.summary.scalar("accuracy", accuracy)

            learning_rate = get_learning_rate(batch)
            tf.summary.scalar("learning_rate", learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        init = tf.global_variables_initializer()
        sess.run(init)
        # sess.run(init, {is_training_pl: True})

        ops = {
            "pointcloud_pl": pointcloud_pl,
            "label_pl": label_pl,
            "is_training_pl": is_training_pl,
            "loss": loss,
            "train_op": train_op,
            "pred": pred,
            "merged": merged,
            "step": batch
        }

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 5 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    is_training = True

    train_file_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_file_idxs)
    for fn in range(len(TRAIN_DATASET)):

        current_data, current_label = provider.loadDataFile(TRAIN_DATASET[train_file_idxs[fn]])
        current_data = current_data[:, 0:NUM_POINT, :]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        current_label = np.squeeze(current_label)
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            pc_data = provider.random_scale_point_cloud(current_data[start_idx:end_idx, :, :])

            pc_data = provider.shift_point_cloud(pc_data)

            feed_dict = {ops['pointcloud_pl']: pc_data,
                         ops['label_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}

            summary, step, _, loss_val, pred_val = sess.run([
                ops["merged"],
                ops["step"],
                ops["train_op"],
                ops["loss"],
                ops["pred"]
            ], feed_dict=feed_dict)

            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct = total_correct + correct
            total_seen = total_seen + BATCH_SIZE
            loss_sum = loss_sum + loss_val

        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops, eval_writer):
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    for fn in range(len(TEST_DATASET)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_DATASET[fn])
        current_data = current_data[:, 0:NUM_POINT, :]
        current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            feed_dict = {ops['pointcloud_pl']: current_data[start_idx:end_idx, :, :],
                         ops['label_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                          ops['loss'], ops['pred']], feed_dict=feed_dict)
            eval_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val * BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i - start_idx] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
