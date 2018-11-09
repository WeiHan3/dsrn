import tensorflow as tf
import util


def dataset(hr_flist,
            scale,
            upsampling_method,
            num_epochs,
            resize=False,
            residual=True):
  """Build the TF sub graph for the inputdata pipeline."""
  with open(hr_flist) as f:
    hr_filename_list = f.read().splitlines()
  with open(lr_flist) as f:
    lr_filename_list = f.read().splitlines()
  filename_queue = tf.train.slice_input_producer(
      [hr_filename_list, lr_filename_list], num_epochs=num_epochs)
  hr_image_file = tf.read_file(filename_queue[0])
  hr_image = tf.image.decode_image(hr_image_file, channels=3)
  hr_image = tf.image.convert_image_dtype(hr_image, tf.float32)
  
  target_scale = tf.random_uniform(shape=[], minval=0.5, maxval=1.0)
  hr_image = _rescale(hr_image, target_scale)
  lr_image = _rescale(hr_image, target_scale/scale)

  if (residual):
    hr_image = _make_residual(hr_image, lr_image, upsampling_method)
  hr_patches0, lr_patches0 = _make_patches(hr_image, lr_image, scale, resize,
                                           upsampling_method)
  hr_patches1, lr_patches1 = _make_patches(
      tf.image.rot90(hr_image),
      tf.image.rot90(lr_image), scale, resize, upsampling_method)
  return tf.concat([hr_patches0, hr_patches1],
                   0), tf.concat([lr_patches0, lr_patches1], 0)


def _rescale(image, target_scale):
  new_shape = tf.to_int32(tf.shape(image)[:2] * target_scale)
  return tf.image.resize_image(image, new_shape, preserve_aspect_ratio=True)

def _make_residual(hr_image, lr_image, upsampling_method):
  """Compute the difference between HR and upsampled LR images"""
  hr_image = tf.expand_dims(hr_image, 0)
  lr_image = tf.expand_dims(lr_image, 0)
  hr_image_shape = tf.shape(hr_image)[1:3]
  res_image = hr_image - util.get_resize_func(upsampling_method)(lr_image,
                                                                 hr_image_shape)
  return tf.reshape(res_image, [hr_image_shape[0], hr_image_shape[1], 3])


def _make_patches(hr_image, lr_image, scale, resize, upsampling_method):
  """Extract patches from images, also apply augmentations"""
  hr_image = tf.stack(_flip([hr_image]))
  lr_image = tf.stack(_flip([lr_image]))
  hr_patches = util.image_to_patches(hr_image)
  if (resize):
    lr_image = util.get_resize_func(upsampling_method)(lr_image,
                                                       tf.shape(hr_image)[1:3])
    lr_patches = util.image_to_patches(lr_image)
  else:
    lr_patches = util.image_to_patches(lr_image, scale)
  return hr_patches, lr_patches


def _flip(img_list):
  flipped_list = []
  for img in img_list:
    flipped_list.append(
        tf.image.random_flip_up_down(
            tf.image.random_flip_left_right(img, seed=0), seed=0))
  return flipped_list
