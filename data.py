import SimpleITK as sitk
import numpy as np
import os
import pickle
from random import shuffle


path = "./dataset"
data_list = ["1","4","5","7","14","070","148"]


class Subjects(object):

  def __init__(self, name, path):
    self.name = name
    self.path = path
    self.get_arrays()

  def __str__(self):
    return "Subject: {}\nData path: {}\n".format(self.name,
      self.path)

  def open_files(self):
    self.flair_img = sitk.ReadImage(os.path.join(self.path,
      self.name, 
      "pre/FLAIR.nii.gz"))
    self.t1_img = sitk.ReadImage(os.path.join(self.path, 
      self.name, 
      "pre/reg_T1.nii.gz"))
    self.ir_img = sitk.ReadImage(os.path.join(self.path, 
      self.name, 
      "pre/reg_IR.nii.gz"))
    self.label_img = sitk.ReadImage(os.path.join(self.path,
      self.name, 
      "segm.nii.gz"))

  def get_arrays(self):
    self.open_files()
    self.flair_array = sitk.GetArrayFromImage(self.flair_img)
    self.t1_array = sitk.GetArrayFromImage(self.t1_img)
    self.ir_array = sitk.GetArrayFromImage(self.ir_img)
    self.label_array = sitk.GetArrayFromImage(self.label_img)
    self.shape = self.flair_array.shape
    self.spacing = self.flair_img.GetSpacing()
    self.origin = self.flair_img.GetOrigin()
    self.direction = self.flair_img.GetDirection()

  def add_dims_3d(self):
    self.label_shape = tuple([1]+list(self.shape)+[11])
    self.shape = tuple([1]+list(self.shape)+[1])
    self.flair_array = np.reshape(self.flair_array, self.shape)
    self.t1_array = np.reshape(self.t1_array, self.shape)
    self.ir_array = np.reshape(self.ir_array, self.shape)
    self.label_array = one_hot_encode(self.label_array, self.label_shape)

  def new_prediction(self):
    self.pred_array = np.zeros(self.label_shape)

  def get_dsc(self):
    pred = np.argmax(self.pred_array, axis=4)
    pred = one_hot_encode(pred, self.label_shape)
    numer = 2*np.sum(pred*self.label_array, axis=(0,1,2,3))
    denom = np.sum(pred,axis=(0,1,2,3)) + \
      np.sum(self.label_array,axis=(0,1,2,3))
    tmp = (numer + denom + 1).astype(np.int32)
    tmp[tmp > 1] = 0
    numer[tmp == 1] = 1
    denom[tmp == 1] = 1
    self.dsc = numer / denom

  def save_prediction(self, patch_size, result_path):
    self.pred_array = np.squeeze(self.pred_array)
    self.pred_array = np.argmax(self.pred_array, axis=3).astype(np.float32)
    self.result_img = sitk.GetImageFromArray(self.pred_array)
    self.result_img.CopyInformation(self.flair_img)
    img_path = os.path.join(result_path, self.name + "_seg.nii.gz")
    sitk.WriteImage(self.result_img, img_path)


def one_hot_encode(array, shape):
  array += 1
  channels = shape[-1]
  array = np.repeat(array, channels)
  array = np.reshape(array, shape)
  array = array / np.arange(1,channels+1)
  array[array > 1] = 0
  array[array < 1] = 0
  return array


def get_files(checkpoint="train/run_1/checkpoints/run_1", 
    train_subjects = 5,
    data_list=data_list,
    shuffle_files=True):
  try:
    files = load_files_order(checkpoint)
  except FileNotFoundError:
    if shuffle_files:
      shuffle(data_list)
    files = {}
    files["train"] = data_list[:train_subjects]
    files["val"] = data_list[train_subjects:]
    save_files_order(files, checkpoint)
  return files


def load_files_order(checkpoint):
  files = {}
  with open(checkpoint, 'rb') as handle:
    files = pickle.load(handle)
    handle.close()
  print('File ' + checkpoint + ' loaded\n')
  return files


def save_files_order(files, checkpoint):
  with open(checkpoint, 'wb') as handle:
    pickle.dump(files, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
  print('File ' + checkpoint + ' saved\n')


def get_objects(files, path=path):
  subjects_dict = {}
  subjects_dict["train"] = []
  subjects_dict["val"] = []
  for key in files.keys():
    n_files = len(files[key])
    print("\n{} {} data {}".format("*"*20,key,"*"*20))
    for n in range(n_files):
      subject = files[key][n]
      subjects_dict[key].append(Subjects(subject,path))
      print("Subject: "+subject)
  print()
  return subjects_dict


def add_extra_dims(subjects_dict, dims=3):
  for key in subjects_dict.keys():
    n_files = len(subjects_dict[key])
    for n in range(n_files):
      subject = subjects_dict[key][n]
      if dims == 3:
        subject.add_dims_3d()


def main():
  files = get_files()
  subjects_dict = get_objects(files)
  subject = subjects_dict["train"][0]
  subject.add_dims_3d()
  subject.new_prediction()
  subject.get_dsc()
  print(subject)
  print(subject.shape)
  print(subject.label_shape)
  print(subject.dsc)
  print("done!")


if __name__ == '__main__':
  main()