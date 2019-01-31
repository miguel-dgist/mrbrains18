import tensorflow as tf
import sys
import os

import config
from models import cnn_3d_segmentation_1, resnet_3d_segmentation_1
from utils import get_loss, batch_norm_3d, get_dsc
from data import get_files, get_objects, add_extra_dims


FLAGS = tf.app.flags.FLAGS


def get_dataset():
  dataset_files = get_files(checkpoint=FLAGS.files_checkpoint, 
    train_subjects=FLAGS.train_subjects)
  dataset = get_objects(dataset_files)
  add_extra_dims(dataset)
  return dataset


def model(inputs):
  if FLAGS.model == "resnet_3d_1":
    net = resnet_3d_segmentation_1(inputs=inputs)
  else:
    net = cnn_3d_segmentation_1(inputs=inputs)
  return net


def build_model(inputs, labels):
  x = batch_norm_3d(inputs=inputs,name="input/batch_norm")
  net = model(x)
  loss = get_loss(labels=labels,
    predictions=net["output"],
    loss_type=FLAGS.loss_type,
    scope=FLAGS.loss_type,
    huber_delta=FLAGS.huber_delta)
  dsc = get_dsc(labels=labels,
    predictions=net["output"])
  net["loss"] = loss
  net["dsc"] = dsc
  return net


def main():
  subjects_dict = get_dataset()
  subject = subjects_dict["train"][0]
  print(subject)
  print(subject.shape)

  x = tf.random_uniform((5,8,24,24,3))
  y = tf.random_uniform((5,8,24,24,11))
  net = build_model(inputs=x, labels=y)
  writer = tf.summary.FileWriter('./tensorboard')
  writer.add_graph(tf.get_default_graph())

  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]="0"

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  network = sess.run(net)
  print(network["output"].shape)
  print("done!")


if __name__ == '__main__':
  main()