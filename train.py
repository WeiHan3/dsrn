import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import util

flags = tf.app.flags
FLAGS = flags.FLAGS

# Task specification.
flags.DEFINE_string('hr_flist', '',
                    'file_list containing the training data.')
flags.DEFINE_integer('scale', '2', 'batch size for training')

# Model and data preprocessing.
flags.DEFINE_string('data_name', '', 'Path to the data specification file.')
flags.DEFINE_string('model_name', '', 'Path to the model specification file.')

flags.DEFINE_string('load_checkpoint', '',
                    'If given, load the checkpoint to initialize model.')
flags.DEFINE_string('output_dir', '',
                    'Path to save the model checkpoint during training.')

# Training hyper parameters
flags.DEFINE_float('learning_rate', '0.001', 'Learning rate.')
flags.DEFINE_integer('batch_size', '32', 'batch size.')
flags.DEFINE_float('ohnm', '1.0', 'percentage of hard negatives')
flags.DEFINE_integer('num_epochs', 3, 'number of epochs')
flags.DEFINE_string('upsampling_method', 'bicubic', 'nn or bicubic')

data = __import__(FLAGS.data_name)
model = __import__(FLAGS.model_name)


def build_data(g):
  """Build the data input pipeline."""
  with tf.device('/cpu:0'):
    with tf.name_scope('data'):
      target_patches, source_patches = data.dataset(
          FLAGS.hr_flist, FLAGS.scale, FLAGS.upsampling_method,
          FLAGS.num_epochs)
      target_batch_staging, source_batch_staging = tf.train.shuffle_batch(
          [target_patches, source_patches],
          FLAGS.batch_size,
          32768,
          8192,
          num_threads=4,
          enqueue_many=True)
  with tf.name_scope('data_staging'):
    stager = data_flow_ops.StagingArea(
        [tf.float32, tf.float32],
        shapes=[[None, None, None, 3], [None, None, None, 3]])
    stage = stager.put([target_batch_staging, source_batch_staging])
    target_batch, source_batch = stager.get()

  return target_batch, source_batch


def build_model(source, target):
  """Build the model graph."""
  with tf.name_scope('model'):
    prediction = model.build_model(
        source, FLAGS.scale, training=True, reuse=False)
    target_cropped = util.crop_center(target, tf.shape(prediction)[1:3])
    tf.summary.histogram('prediction', prediction)
    tf.summary.histogram('groundtruth', target)
  return prediction, target_cropped


def build_loss(prediction, target_cropped):
  with tf.name_scope('l2_loss'):
    if FLAGS.ohnm < 1.0:
      pixel_loss = tf.reduce_sum(
          tf.square(tf.subtract(target_cropped_batch, predict_batch)), 3)
      raw_loss = tf.reshape(pixel_loss, [-1])
      num_ele = tf.size(raw_loss)
      num_negative = tf.cast(
          tf.to_float(num_ele) * tf.constant(FLAGS.ohnm), tf.int32)
      hard_negative, _ = tf.nn.top_k(raw_loss, num_negative)
      avg_loss = tf.losses.mean_squared_error(target_cropped_batch,
                                              predict_batch)

      hard_loss = tf.reduce_mean(hard_negative)
      tf.summary.scalar('training_l2_loss', avg_loss)
      tf.summary.scalar('training_hard_l2_loss', hard_loss)
      loss = hard_loss
    else:
      if FLAGS.precision > 0:
        loss = tf.reduce_mean(
            tf.square(
                tf.nn.relu(
                    tf.abs(target_cropped_batch - predict_batch) -
                    FLAGS.precision / tf.uint8.max)))
      else:
        loss = tf.losses.mean_squared_error(target_cropped_batch, predict_batch)

      tf.summary.scalar('training_l2_loss', loss)


def build_trainer(loss):
  with tf.name_scope('train'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gvs]
    optimizer = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    merged_summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

  return global_step, optimizer, merged_summary_op, init


def prepare_directories(outdir):

  def make_dir(d):
    if not os.path.exists(d):
      os.mkdir(d)

  make_dir(outdir)
  ckpt_dir = os.path.join(outdir, "train")
  summary_dir = os.path.join(outdir, "summary")
  make_dir(ckpt_dir)
  make_dir(summary_dir)
  return ckpt_dir, summary_dir


def main():
  g = tf.Graph()
  with g.as_default():
    src, tgt = build_data()
    loss = build_model(src, tgt)
    global_step, train_op, summary_op, init_op = build_trainer(loss)

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.95

  # Prepare output dir.
  ckpt_dir, summary_dir = prepare_directories(FLAGS.output_dir)

  with tf.Session(graph=g, config=config) as sess:
    train_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    sess.run(init_op)
    if tf.gfile.Exists(FLAGS.load_checkpoint):
      print('Loading model from %s', FLAGS.load_checkpoint)
      saver.restore(sess, FLAGS.load_checkpoint)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      sess.run(stage)
      while not coord.should_stop():
        _, _, step, training_loss, train_summary = sess.run([
            stage,
            train_op,
            step,
            loss,
            summary_op,
        ])
        print('Training at step %d, loss=%f', step, training_loss)
        train_writer.add_summary(train_summary, step)
        if (step % 1000 == 0):
          saver.save(sess, ckpt_dir, global_step=global_step)
    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      coord.request_stop()
      saver.save(sess, FLAGS.model_file_out, global_step=global_step)


if __name__ == '__main__':
  tf.app.run()
