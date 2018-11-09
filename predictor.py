from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from google3.protobuf import text_format
from lingvo.core import inference_graph_pb2
from lingvo.core import py_utils

import os
import skimage
import skimage.io

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint', '',
                            """Path to model checkpoint.""")
tf.app.flags.DEFINE_string('inference_graph', '',
                            """Path to model inference_graph def.""")
tf.app.flags.DEFINE_string('image_path', '',
                            """Path to input image in a format supported by tf.image.decode_image.""")
tf.app.flags.DEFINE_string('output_dir', '',
                            """Output directory, results will be written as png files.""")


def LoadInferenceGraph(path):
  inference_graph = inference_graph_pb2.InferenceGraph()
  with tf.gfile.Open(path, "r") as f:
    text_format.Parse(f.read(), inference_graph)
  return inference_graph


class Predictor(object):
  def __init__(self,
               inference_graph,
               checkpoint):
    """Initialize the predictor,

    Args:
      inferece_graph: a text file containing an inference_graph proto.
      checkpoint: actual model checkpoint (without '.meta').
    """
    inference_graph = LoadInferenceGraph(inference_graph)
    self._checkpoint = checkpoint
    self._graph = tf.Graph()
    with self._graph.as_default():
      self._saver = tf.train.Saver(saver_def=inference_graph.saver_def)
      with tf.device("cpu:0" % "cpu"):
        tf.import_graph_def(inference_graph.graph_def, name="")
      self._graph.finalize()

    subgraph = inference_graph.subgraphs['default']
    assert 'img_str' in subgraph.feeds
    assert 'hr_image' in subgraph.fetches

    self._sess = tf.Session(graph=self._graph)
    self._saver.restore(self._sess, self._checkpoint)

  def Run(self, image_name):
    """Runs predictor
    Args:
      image_name: full path to the image file.

    Returns:
      A numpy array of the output image.
    """
    img_raw_str = tf.gfile.Open(image_name, 'rb').read()
    hr_output = self._sess.run('hr_image', feed_dict={'img_str': [img_raw_str]})
    return hr_output[0]


def main(_):
  predictor = Predictor(FLAGS.inference_graph,
                        FLAGS.checkpoint)
  out_image = predictor.Run(FLAGS.image_path)
  image_name = os.path.splitext(os.path.basename(FLAGS.image_path))[0]
  out_image_path = os.path.join(FLAGS.out_dir, image_name + '.png')

  skimage.io.imsave(out_image_path, out_image)


if __name__ == "__main__":
  tf.app.run(main)
