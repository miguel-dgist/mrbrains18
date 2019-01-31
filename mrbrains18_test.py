import tensorflow as tf
import os
import numpy as np

from mrbrains18 import build_model, get_dataset
from utils import re_arrange_array

FLAGS = tf.app.flags.FLAGS


def test(dataset, patch_size, checkpoint_number):
  
  pz = patch_size[0]
  py = patch_size[1]
  px = patch_size[2]

  x_flair = tf.placeholder(dtype=tf.float32, shape=[None,pz,py,px,1])
  x_t1 = tf.placeholder(dtype=tf.float32, shape=[None,pz,py,px,1])
  x_ir = tf.placeholder(dtype=tf.float32, shape=[None,pz,py,px,1])
  y_gt = tf.placeholder(dtype=tf.float32, shape=[None,pz,py,px,11])

  x = tf.concat(values=(x_flair,x_t1,x_ir), axis=4, name="input/concat")

  net = build_model(inputs=x, labels=y_gt)

  sess = tf.Session()

  saver = tf.train.Saver()
  saver.restore(sess, FLAGS.checkpoint_path+'-'+checkpoint_number)

  print()
  print("Checkpoint: {}".format(checkpoint_number))
  test_size = len(dataset["val"])
  avg_dsc = 0
  for m in range(test_size):
    subject = dataset["val"][m]
    subject.new_prediction()
    shape = subject.shape
    for z in range(0,shape[1]-pz,int(pz/2)):
      for y in range(0,int(py/2)+1,int(py/2)):
        y2 = y + int((shape[2]-y)/py)*py
        for x in range(0,int(px/2)+1,int(px/2)):
          x2 = x + int((shape[3]-x)/px)*px
          tmp_flair = subject.flair_array[:,z:z+pz,y:y2,x:x2]
          tmp_shape = list(tmp_flair.shape)
          tmp_flair = re_arrange_array(tmp_flair, tmp_shape, "input")
          tmp_t1 = subject.t1_array[:,z:z+pz,y:y2,x:x2]
          tmp_t1 = re_arrange_array(tmp_t1, tmp_shape, "input")
          tmp_ir = subject.ir_array[:,z:z+pz,y:y2,x:x2]
          tmp_ir = re_arrange_array(tmp_ir, tmp_shape, "input")
          tmp_label = sess.run(net["output"], feed_dict={\
            x_flair: tmp_flair,
            x_t1: tmp_t1,
            x_ir: tmp_ir})
          tmp_shape[-1] = subject.pred_array.shape[-1]
          tmp_label = re_arrange_array(tmp_label, tmp_shape, "output")
          subject.pred_array[:,z:z+pz,y:y2,x:x2] += tmp_label
    subject.pred_array = subject.pred_array/8
    subject.get_dsc()
    avg_dsc += subject.dsc
    test_status = "{}\nBackground:\t\t{:0.4f}\n" + \
      "Cortical gray matter:\t{:0.4f}\n" + \
      "Basal ganglia:\t\t{:0.4f}\n" + \
      "White matter:\t\t{:0.4f}\n" + \
      "White matter lesions:\t{:0.4f}\n" + \
      "Cerebrospinal fluid:\t{:0.4f}\n" + \
      "Ventricles:\t\t{:0.4f}\n" + \
      "Cerebellum:\t\t{:0.4f}\n" + \
      "Brain stem:\t\t{:0.4f}\n" + \
      "Infarction:\t\t{:0.4f}\n" + \
      "Other:\t\t\t{:0.4f}\n"
    print(test_status.format(subject.name,subject.dsc[0],
      subject.dsc[1],
      subject.dsc[2],
      subject.dsc[3],
      subject.dsc[4],
      subject.dsc[5],
      subject.dsc[6],
      subject.dsc[7],
      subject.dsc[8],
      subject.dsc[9],
      subject.dsc[10]))
    del subject.pred_array

  avg_dsc = avg_dsc/test_size
  avg_dsc = np.mean(avg_dsc[1:-2])

  text = "Checkpoint: {} - average dsc = {:0.3f}"
  print(text.format(checkpoint_number, avg_dsc))
  print()
  sess.close()
  tf.reset_default_graph()


def main():
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.cuda_device
  dataset = get_dataset()
  patch_size = list(map(int, FLAGS.patch_size.split(",")))
  checkpoints = FLAGS.test_checkpoints.split(",")
  for n in range(len(checkpoints)):
    checkpoint_number = checkpoints[n]
    test(dataset, patch_size, checkpoint_number)
    print()
  print("done!")


if __name__ == '__main__':
  main()