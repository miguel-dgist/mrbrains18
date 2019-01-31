import tensorflow as tf
import numpy as np


def get_loss(labels,predictions,loss_type,scope=None,**kwargs):
  """ Calculates a compensated loss for all 11 labels.
      The losses available are:
        * absolute_difference
        * mean_squared_error
        * log_loss
        * huber_loss - requires huber_delta
  """
  if loss_type == "absolute_difference":
    loss_func = lambda x,y,z: tf.losses.absolute_difference(labels=x,
      predictions=y,
      weights=z,
      reduction=tf.losses.Reduction.NONE)
  elif loss_type == "mean_squared_error":
    loss_func = lambda x,y,z: tf.losses.mean_squared_error(labels=x,
      predictions=y,
      weights=z,
      reduction=tf.losses.Reduction.NONE)
  elif loss_type == "log_loss":
    loss_func = lambda x,y,z: tf.losses.log_loss(labels=x,
      predictions=y,
      weights=z,
      reduction=tf.losses.Reduction.NONE)
  elif loss_type == "huber_loss":
    loss_func = lambda x,y,z: tf.losses.huber_loss(labels=x,
      predictions=y,
      weights=z,
      delta=kwargs["huber_delta"],
      reduction=tf.losses.Reduction.NONE)
  else:
    print("*"*20)
    print("Not valid loss function was defined")
    return tf.zeros((1,))
  with tf.name_scope(scope,"loss"):
    shape = tf.shape(labels)
    axes = tf.range(tf.shape(shape)[0]-1)
    loss_1 = loss_func(labels,predictions,labels)
    nonzero = tf.reduce_sum(labels, axis=axes)+1e-9
    loss_1 = tf.reduce_sum(loss_1,axis=axes)/nonzero
    loss_1 = tf.reduce_mean(loss_1)
    loss_2 = loss_func(labels,predictions,1-labels)
    nonzero = tf.reduce_sum(1-labels, axis=axes)+1e-9
    loss_2 = tf.reduce_sum(loss_2,axis=axes)/nonzero
    loss_2 = tf.reduce_mean(loss_2)
    loss = (loss_1 + loss_2)/2
  return loss


def get_dsc(labels,predictions,scope=None):
  with tf.name_scope(scope, "dsc"):
    shape = tf.shape(labels)
    axes_1 = tf.shape(shape)[0]-1
    axes = tf.range(tf.shape(shape)[0]-1)
    pred = tf.argmax(predictions, axis=axes_1)
    pred = tf.one_hot(indices=pred, 
      depth=shape[-1],
      on_value=1,
      off_value=0,
      axis=-1)
    pred = tf.cast(pred, tf.float32)
    numer = 2*tf.reduce_sum(labels*pred, axis=axes)
    denom = tf.reduce_sum(pred,axis=axes)+tf.reduce_sum(labels,axis=axes)
    equal = tf.cast(tf.equal(numer,denom), tf.float32)
    dsc = (numer+equal) / (denom+equal)
  return dsc


def batch_norm_3d(inputs, name=None):
  with tf.name_scope(name, "batch_norm"):
    batch = tf.transpose(inputs, perm=[1,2,3,0,4])
    mean, var = tf.nn.moments(batch,axes=[0,1,2])
    batch = tf.nn.batch_normalization(batch, 
      mean=mean, 
      variance=var,
      offset=0,
      scale=1,
      variance_epsilon=1e-9)
    batch = tf.transpose(batch, perm=[3,0,1,2,4])
  return batch


def re_arrange_array(array, new_shape, mode):
  if mode == "input":
    shape = (np.array(array.shape)/24).astype(np.int32)
    array = np.transpose(array, (0,2,3,1,4))
    array = np.reshape(array, (shape[2],24,shape[3]*24,8,1))
    array = np.transpose(array, (0,2,1,3,4))
    array = np.reshape(array, (shape[2]*shape[3],24,24,8,1))
    array = np.transpose(array, (0,3,2,1,4))
  elif mode == "output":
    shape = (np.array(array.shape)).astype(np.int32)
    new_shape = (np.array(new_shape)).astype(np.int32)
    array = np.transpose(array, (0,3,2,1,4))
    array = np.reshape(array, (int(shape[0]/(new_shape[3]/24)), \
      new_shape[3],24,8,new_shape[4]))
    array = np.transpose(array, (0,2,1,3,4))
    array = np.reshape(array, (1,new_shape[2],new_shape[3],8,new_shape[4]))
    array = np.transpose(array, (0,3,1,2,4))
  return array
