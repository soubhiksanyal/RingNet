"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about RingNet is available at https://ringnet.is.tue.mpg.de.

All rights reserved.
based on github.com/akanazawa/hmr
"""
# RingNet Inference for single image.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from os.path import exists

class RingNet_inference(object):
    def __init__(self, config, sess=None):
        self.config = config
        self.load_path = config.load_path
        if not config.load_path:
            raise Exception(
                "provide a pretrained model path"
            )
        if not exists(config.load_path + '.index'):
            print('%s couldnt find..' % config.load_path)
            import ipdb
            ipdb.set_trace()

        # Data
        self.batch_size = config.batch_size
        self.img_size = config.img_size
        self.data_format = config.data_format
        input_size = (self.batch_size, self.img_size, self.img_size, 3)
        self.images_pl = tf.placeholder(tf.float32, shape=input_size, name='input_images')

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        # Load graph.
        self.saver = tf.train.import_meta_graph(self.load_path+'.meta')
        self.graph = tf.get_default_graph()
        self.prepare()


    def prepare(self):
        print('Restoring checkpoint %s..' % self.load_path)
        self.saver.restore(self.sess, self.load_path)


    def predict(self, images, get_parameters=False):
        """
        images: batch_size, img_size, img_size, 3 # Here for inference the batch size is always set to 1
        Preprocessed to range [-1, 1]
        """
        results = self.predict_dict(images)
        if get_parameters:
            return results['vertices'], results['parameters']
        else:
            return results['vertices']


    def predict_dict(self, images):
        """
        Runs the model with images.
        """
        images_ip = self.graph.get_tensor_by_name(u'input_images_1:0')
        params = self.graph.get_tensor_by_name(u'add_2:0')
        verts = self.graph.get_tensor_by_name(u'Flamenetnormal_2/Add_9:0')
        feed_dict = {
            images_ip: images,
        }
        fetch_dict = {
            'vertices': verts,
            'parameters': params,
        }
        results = self.sess.run(fetch_dict, feed_dict)
        tf.reset_default_graph()
        return results
