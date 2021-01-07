import tensorflow as tf
import skimage
import skimage.io
import skimage.color
import skimage.measure

flags = tf.app.flags
FLAGS = flags.FLAGS

# Task specification.
flags.DEFINE_string('hr_flist', '',
                    'file_list containing the training data.')
flags.DEFINE_string('prediction_dir', '',
                    'directory containing the predicted images.')


def compute_psnr(prediction, ground_truth):
  pred_y = skimage.color.rgb2ycbcr(prediction)[:,:,0:1]
  gt_y = skimage.color.rgb2ycbcr(ground_truth)[:,:,0:1]
  return skimage.measure.compare_psnr(pred_y, gt_y, data_range=255)

def main():
  flist = open(FLAGS.hr_flist, 'r').read().splitlines()
  total_images = 0
  total_psnr = .0
  for fname in flist:
    pred_fname = os.path.join(FLAGS.prediction_dir,
                              os.path.basename(fname))
    gt_image = skimage.io.imread(fname)
    pred_image = skimage.io.imread(pred_fname)
    psnr = compute_psnr(pred_image, gt_image)
    print("Image name: %s, PSNR=%f", os.path.basename(fname),
          psnr)
    total_images += 1
    total_psnr += psnr

  print("Average PSNR is %f", total_psnr/total_images)

if __name__ == '__main__':
  tf.app.run()
