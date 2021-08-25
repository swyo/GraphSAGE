import sys
import unittest

import numpy as np
import tensorflow as tf

sys.path.append('..')
from graphsage.models import SAGEInfo
from graphsage.utils import load_data
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.supervised_train import SupervisedGraphsage
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.supervised_train import construct_placeholders


class TestMinibatch(unittest.TestCase):
    def _generate_minibatch(self, prefix='../example_data/toy-ppi', batch_size=512, max_degree=128):
        G, features, id_map, context_pairs, class_map = load_data(
            prefix, normalize=True, load_walks=True
        )
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
        else:
            num_classes = len(set(class_map.values()))
        if not features is None:
            # pad with dummy zero vector
            features = np.vstack([features, np.zeros((features.shape[1],))])
        placeholders = construct_placeholders(num_classes)
        return NodeMinibatchIterator(
            G, id_map, placeholders, class_map, num_classes,
            batch_size=batch_size, max_degree=max_degree, context_pairs=context_pairs
        ), features

    # @unittest.skip("Skip Init")
    def test01_init(self):
        minibatch, _ = self._generate_minibatch()
        self.assertIsInstance(
            minibatch, NodeMinibatchIterator, 'The class of minibatch is not accord with NodeMinibatchIterator'
        )
    
    # @unittest.skip("Skip Iteration")
    def test02_iteration(self):
        minibatch, _ = self._generate_minibatch()
        try:
            for epoch in range(1):
                minibatch.shuffle()
                while not minibatch.end():
                    feed_dict, _labels = minibatch.next_minibatch_feed_dict()
                    _batch_size, batch, labels = feed_dict.values()
                    self.assertLessEqual(_batch_size, minibatch.batch_size)
                    self.assertEqual(len(batch), _batch_size)
                    self.assertEqual(labels.shape, (_batch_size, minibatch.num_classes))
        except Exception:
            self.assertTrue(False, 'Iterate an epoch does not finished')

    def test03_sample(self):
        minibatch, features = self._generate_minibatch()
        adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
        sampler = UniformNeighborSampler(adj_info)
        samples_1, samples_2, dim = 25, 10, 128
        layer_infos = [
            SAGEInfo("node", sampler, samples_1, dim),
            SAGEInfo("node", sampler, samples_2, dim)
        ]
        model = SupervisedGraphsage(
            minibatch.num_classes, minibatch.placeholders, features,
            adj_info, minibatch.deg, layer_infos, 
            model_size='small', sigmoid_loss=True,
            identity_dim=0, logging=True)
        samples_once = sampler((minibatch.placeholders['batch'], layer_infos[0].num_samples))
        samples_all_hops, support_sizes = model.sample(minibatch.placeholders['batch'], layer_infos)
        num_samples = [layer_info.num_samples for layer_info in layer_infos]
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
            minibatch.shuffle()
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
            _batch_size, batch, labels = feed_dict.values()
            # check for sampling once.
            out = sess.run(samples_once, feed_dict=feed_dict)
            self.assertEqual(out.shape, (minibatch.batch_size, layer_infos[0].num_samples))
            adj_list = minibatch.adj[np.array(batch)]
            for i, o in enumerate(out):
                self.assertTrue(~bool(set(o) - set(adj_list[i])))
            # check for sampling multi-hops
            outs = sess.run(samples_all_hops, feed_dict=feed_dict)
            for i, _samples_in_hop in enumerate(outs):
                self.assertEqual(len(_samples_in_hop), support_sizes[i] * minibatch.batch_size)
                if i == 0:
                    self.assertTrue(np.array_equal(np.array(_samples_in_hop), batch))
