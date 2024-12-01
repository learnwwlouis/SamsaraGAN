from glob import glob
import numpy as np
import scipy.misc


def load_train_data(dataA, dataB, idx, image_size, batch_size):
  batch_files  = zip(dataA[idx*batch_size : (idx+1)*batch_size],
                     dataB[idx*batch_size : (idx+1)*batch_size])
  batch_images = [load_data(batch_file, image_size) for batch_file in batch_files]
  batch_images = np.array(batch_images).astype(np.float32)
  batch_files = []
  return batch_images

def load_data(image_path, image_size, flip=True):
  img_A, img_B = load_image(image_path)
  img_A, img_B = preprocess_A_and_B(img_A, img_B, fine_size=image_size,flip=flip)

  # normailize [-1.0, +1.0]
  img_A = img_A/127.5 - 1.
  img_B = img_B/127.5 - 1.

  img_AB = np.concatenate((img_A, img_B), axis=2)

  return img_AB

def load_image(image_path):
  img_A = imread(image_path[0])
  img_B = imread(image_path[1])
  return img_A, img_B

def preprocess_A_and_B(img_A, img_B, fine_size=256, flip=True):
  img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
  img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])

  if flip and np.random.random() > 0.5:
    img_A = np.fliplr(img_A)
    img_B = np.fliplr(img_B)

  return img_A, img_B

def imread(path, is_grayscale=False):
  if (is_grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path, mode='RGB').astype(np.float)
