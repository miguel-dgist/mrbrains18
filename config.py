import tensorflow as tf


tf.app.flags.DEFINE_string('data_dir', '/home/miguel/Datasets/MRBrainS18',
  """ Directory where to find the dataset.""")

tf.app.flags.DEFINE_string('files_checkpoint', 
  '/home/miguel/Downloads/mrbs18_obj_ckpt.p', 
  """ Checkpoint to keep track of training, and validation
      subjects distribution""")

tf.app.flags.DEFINE_integer('train_subjects', 5,
  """ Number of training subjects out of 7.""")

tf.app.flags.DEFINE_string('model', 'cnn_3d_1',
  """ Model selection """)

tf.app.flags.DEFINE_string('loss_type', 'log_loss',
  """ Loss selection """)

tf.app.flags.DEFINE_float('huber_delta', 0.3,
  """ Delta parameter for Huber loss """)

tf.app.flags.DEFINE_string('patch_size', "8,24,24", 
  """ Height and width of the image """)

tf.app.flags.DEFINE_string('cuda_device', '0',
  """ Select CUDA device to run the model.""")

tf.app.flags.DEFINE_integer('batch_size', 128,
  """ Number of images to be run at the same time.""")

tf.app.flags.DEFINE_integer('keep_checkpoint', 6,
  """ Keep a checkpoint every specified number of hours """)

tf.app.flags.DEFINE_integer('max_steps', 500000,
  """ Number of repetitions during training.""")

tf.app.flags.DEFINE_integer('steps_to_val', 2000,
  """ Number of steps that the validation dataset is run.""")

tf.app.flags.DEFINE_float('learning_rate', 1e-4,
  """ Starting learning rate.""")

tf.app.flags.DEFINE_integer('steps_to_learning_rate_update', 20000,
  """ Number of steps to update the learning rate.""")

tf.app.flags.DEFINE_float('learning_rate_decrease', 0.1,
  """ Decrease factor in the learning rate.""")

tf.app.flags.DEFINE_integer('steps_to_save_checkpoint', 1000,
  """ Number of steps to save a checkpoint.""")

tf.app.flags.DEFINE_string('checkpoint_path', 
  '/home/miguel/Downloads/mrbs_3d_seg',
  """ Path to save training checkpoint """)

tf.app.flags.DEFINE_string('test_checkpoints', 
  '1000,2000',
  """ Checkpoints for every test model """)


# ******************* Only inference *******************

tf.app.flags.DEFINE_string('flair_path', 
  '/input/pre/FLAIR.nii.gz',
  """ Path to load the Flair type image """)

tf.app.flags.DEFINE_string('t1_path', 
  '/input/pre/reg_T1.nii.gz',
  """ Path to load the T1 type image """)

tf.app.flags.DEFINE_string('ir_path', 
  '/input/pre/reg_IR.nii.gz',
  """ Path to load the T1 type image """)

tf.app.flags.DEFINE_string('result_path', 
  '/output/result.nii.gz',
  """ Path to save the resulting segmentation """)

tf.app.flags.DEFINE_string('model_path', 
  '/src/inference/mrbs_3d_seg',
  """ Path to load checkpoint """)