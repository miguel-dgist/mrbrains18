import tensorflow as tf
import numpy as np
import os

def encoder(inputs, 
  scope,
  levels,
  channels,
  number_of_units,
  type_of_layer,
  pool_strides):
  with tf.variable_scope(scope):
    net = {}
    net[scope+"_level_0"] = inputs
    for level in range(levels):
      net[scope+"_level_"+str(level)] = get_layer( \
        net=net[scope+"_level_"+str(level)],
        scope="level_"+str(level),
        type_of_layer=type_of_layer,
        number_of_units=number_of_units[level],
        channels=channels[level])
      if level < (levels-1):
        net[scope+"_level_"+str(level+1)] = tf.layers.average_pooling3d(\
          net[scope+"_level_"+str(level)],
          pool_size=pool_strides[level],
          strides=pool_strides[level],
          padding="same",
          name="level_"+str(level+1)+"/pool")
    return net


def transition_layer(net,
  input_scope,
  scope,
  levels,
  channels):
  with tf.variable_scope(scope):
    for level in range(levels-1):
      net[scope+"_level_"+str(level)] = get_layer( \
        net=net[input_scope+"_level_"+str(level)],
        scope="level_"+str(level),
        type_of_layer="cnn",
        number_of_units=1,
        channels=channels[level])
    net[scope+"_level_"+str(levels-1)] = \
      net[input_scope+"_level_"+str(levels-1)]
  return net


def decoder(net,
  input_scope,
  scope,
  levels,
  channels,
  number_of_units,
  type_of_layer,
  pool_strides):
  with tf.variable_scope(scope):
    net[scope+"_level_"+str(levels-1)] = net[input_scope+"_level_"+str(levels-1)]
    for level in range(levels-2,-1,-1):
      net[scope+"_level_"+str(level)] = tf.layers.conv3d_transpose(\
        inputs=net[scope+"_level_"+str(level+1)],
        filters=channels[level+1],
        kernel_size=pool_strides[level],
        strides=pool_strides[level],
        padding="same",
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
        bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
        name="level_"+str(level)+"/upconv")
      with tf.name_scope("level_"+str(level)+"/batch_norm"):
        net[scope+"_level_"+str(level)] = \
          batch_norm(net[scope+"_level_"+str(level)])
      net[scope+"_level_"+str(level)] = tf.concat(\
        values=(net[scope+"_level_"+str(level)],
          net[input_scope+"_level_"+str(level)]),
        axis=4,
        name="level_"+str(level)+"/concat")
      net[scope+"_level_"+str(level)] = get_layer( \
        net=net[scope+"_level_"+str(level)],
        scope="level_"+str(level),
        type_of_layer=type_of_layer,
        number_of_units=number_of_units[level],
        channels=channels[level])
  return net


def output_layer(net,
  input_scope,
  scope):
  with tf.variable_scope(scope):
    net[scope] = tf.layers.conv3d(inputs=net[input_scope+"_level_0"],
      filters=11,
      kernel_size=1,
      strides=1,
      padding="same",
      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
      bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
      name="conv")
    net[scope] = tf.nn.softmax(logits=net[scope],
      axis=4,
      name="softmax")
  return net


def get_layer(net,
  scope,
  type_of_layer,
  number_of_units,
  channels):
  if type_of_layer == "cnn":
    for n in range(number_of_units):
      net = tf.layers.conv3d(inputs=net,
        filters=channels,
        kernel_size=3,
        strides=1,
        padding="same",
        dilation_rate=1,
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
        bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
        name=scope+"/conv_"+str(n))
      with tf.name_scope(scope+"/batch_norm_"+str(n)):
        net = batch_norm(net)
  elif type_of_layer == "resnet":
    net = tf.layers.conv3d(inputs=net,
      filters=channels,
      kernel_size=3,
      strides=1,
      padding="same",
      dilation_rate=1,
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
      bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
      name=scope+"/conv")
    with tf.name_scope(scope+"/batch_norm"):
      net = batch_norm(net)
    for n in range(number_of_units):
      with tf.variable_scope(scope+"/resnet_module_"+str(n)):
        tmp = net
        for _ in range(2):
          net = tf.layers.conv3d(inputs=net,
            filters=channels,
            kernel_size=3,
            strides=1,
            padding="same",
            dilation_rate=1,
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
            bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))
          net = batch_norm(net)
        net += tmp
        net = batch_norm(net)
  return net


def batch_norm(net):
  net = tf.transpose(net, 
    perm=[1,2,3,0,4])
  mean, var = tf.nn.moments(net, 
    axes=[0,1,2])
  net = tf.nn.batch_normalization(net,
    mean=mean,
    variance=var,
    offset=0,
    scale=1,
    variance_epsilon=1e-9)
  net = tf.transpose(net, 
    perm=[3,0,1,2,4])
  return net


def cnn_3d_segmentation(inputs,
  scope,
  levels,
  channels,
  encoder_units,
  decoder_units,
  pool_strides):
  transition_channels = list((np.array(channels)*0.25).astype(np.int32))
  with tf.variable_scope(scope):
    net = encoder(inputs=inputs,
      scope="encoder",
      levels=levels,
      channels=channels,
      number_of_units=encoder_units,
      type_of_layer="cnn",
      pool_strides=pool_strides)
    net = transition_layer(net=net,
      input_scope="encoder",
      scope="transition",
      levels=levels,
      channels=transition_channels)
    net = decoder(net=net,
      input_scope="transition",
      scope="decoder",
      levels=levels,
      channels=channels,
      number_of_units=decoder_units,
      type_of_layer="cnn",
      pool_strides=pool_strides)
    net = output_layer(net=net,
      input_scope="decoder",
      scope="output")
  return net


def resnet_3d_segmentation(inputs,
  scope,
  levels,
  channels,
  encoder_units,
  decoder_units,
  pool_strides):
  transition_channels = list((np.array(channels)*0.25).astype(np.int32))
  with tf.variable_scope(scope):
    net = encoder(inputs=inputs,
      scope="encoder",
      levels=levels,
      channels=channels,
      number_of_units=encoder_units,
      type_of_layer="resnet",
      pool_strides=pool_strides)
    net = transition_layer(net=net,
      input_scope="encoder",
      scope="transition",
      levels=levels,
      channels=transition_channels)
    net = decoder(net=net,
      input_scope="transition",
      scope="decoder",
      levels=levels,
      channels=channels,
      number_of_units=decoder_units,
      type_of_layer="cnn",
      pool_strides=pool_strides)
    net = output_layer(net=net,
      input_scope="decoder",
      scope="output")
  return net


def cnn_3d_segmentation_1(inputs):
  net = cnn_3d_segmentation(inputs=inputs,
    scope="cnn_3d_1",
    levels=3,
    channels=[64,128,256],
    encoder_units=[3,4,5],
    decoder_units=[2,2],
    pool_strides=[[2,2,2],[1,2,2]])
  return net


def resnet_3d_segmentation_1(inputs):
  net = resnet_3d_segmentation(inputs=inputs,
    scope="resnet_3d_1",
    levels=3,
    channels=[64,128,256],
    encoder_units=[2,3,4],
    decoder_units=[2,2],
    pool_strides=[[2,2,2],[1,2,2]])
  return net


def main():
  x = tf.random_uniform((5,8,24,24,3))
  net = cnn_3d_segmentation_1(inputs=x)
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