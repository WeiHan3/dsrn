import tensorflow as tf
import util
from functools import partial


def _unroll_rnn(x, x_name, T, rnn_graph, average_on_relu, reuse):
  """
  Construct the TF graph for ann defined by its nodes and edges.

  Input:
    x: the input tensor
    rnn_graph: [node_names, edges]
      edges are tuples of <from_node_name, to_node_name, edge_func, delay>.
    average_on_relu: if true, average instead of sum will be used
      to combine outputs from multiple edges.

  Output:
      all tensors: the output tensor at every time step.
  """
  node_names, edges = rnn_graph
  all_tensors = [{} for t in range(T)]

  # Counting the number of input tensors of a node.
  # only useful if average_on_relu=True.
  counter = [{} for t in range(T)]
  all_tensors[0][x_name] = x
  counter[0][x_name] = 1

  for t in range(T):
    # Unroll at time step t.
    for s_name in all_tensors[t]:
      # Finalize state, apply relu and scale.
      in_tensor = all_tensors[t][s_name]
      in_tensor = tf.nn.relu(in_tensor)
      if average_on_relu and counter[t][s_name] > 1:
        in_tensor = in_tensor / counter[t][s_name]
      all_tensors[t][s_name] = in_tensor

      # Look at s_name, find all it's successors.
      for edge in edges:
        e_name1, e_name2, e_fun, e_delay = edge
        if s_name == e_name1:
          if e_delay + t < T:
            try:
              out_tensor = e_fun(in_tensor, reuse=reuse)
            except ValueError:
              # Sharing if exists.
              out_tensor = e_fun(in_tensor, reuse=True)

            if e_name2 in all_tensors[e_delay + t]:
              all_tensors[e_delay +
                          t][e_name2] = all_tensors[e_delay +
                                                    t][e_name2] + out_tensor
              counter[e_delay + t][e_name2] += 1
            else:
              all_tensors[e_delay + t][e_name2] = out_tensor
              counter[e_delay + t][e_name2] = 1
  return all_tensors


# Commonly used edge link functions.
def _conv_upsample(_input, reuse):
  """Conv layer that upsamples the input."""
  output = tf.layers.conv2d_transpose(
      _input,
      hidden_size,
      2,
      strides=2,
      activation=None,
      name='up',
      reuse=reuse)
  return output


def _downsample(_input, reuse):
  """Down-sample via average pooling."""
  output = tf.nn.avg_pool(
      _input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
  return output


def _conv_link(_input, name, reuse):
  with tf.variable_scope(name):
    _tmp = tf.layers.conv2d(
        _input,
        hidden_size,
        3,
        padding='SAME',
        activation=tf.nn.relu,
        name='conv1',
        reuse=reuse)
    output = tf.layers.conv2d(
        _tmp,
        hidden_size,
        3,
        padding='SAME',
        activation=None,
        name='conv2',
        reuse=reuse)
  return output


def _skip_link(_input, reuse):
  return _input


def build_model(x, scale, training, reuse):
  hidden_size = 128
  use_average_out = True
  T = 7

  # Define the RNN in its recurrent form.
  # s1: low-res state. s2: high-res state.
  node_names = ['s1', 's2']
  edges = []
  edges.append(['s1', 's1', _skip_link, 1])
  edges.append(['s1', 's1', partial(_conv_link, name='s1s1'), 1])
  edges.append(['s1', 's2', partial(_conv_upsample, name='s1s2'), 1])
  edges.append(['s2', 's1', partial(_downsample, name='s2s1'), 1])
  edges.append(['s2', 's2', _skip_link, 1])
  edges.append(['s2', 's2', partial(_conv_link, name='s2s2'), 1])

  # InputNet
  with tf.variable_scope('InputNet'):
    s1_0 = tf.layers.conv2d(
        x,
        hidden_size / 2,
        3,
        activation=tf.nn.relu,
        name='in_0',
        padding='same',
        reuse=reuse)
    s1_0 = tf.layers.conv2d(
        s1_0,
        hidden_size,
        3,
        activation=tf.nn.relu,
        name='in_1',
        padding='same',
        reuse=reuse)

  # Unrolled Recurrent Net
  with tf.variable_scope('RecurrentNet'):
    full_net_states = _unroll_rnn(
        s1_0, 's1', T, (node_names, edges), average_on_relu=False, reuse=reuse)

  # OutputNets
  with tf.variable_scope('OutputNet'):
    if use_average_out:
      all_out_states = [
          full_net_states[t]['s2'] for t in range(T)
          if 's2' in full_net_states[t]
      ]
      pre_out = tf.add_n(all_out_states) / len(all_out_states)

    else:
      pre_out = full_net_states[T - 1]['s2']

  # Final output prediction.
  out = tf.layers.conv2d(
      pre_out, 3, 1, activation=None, name='out', reuse=reuse)
  return out
