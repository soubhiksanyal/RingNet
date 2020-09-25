"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about RingNet is available at https://ringnet.is.tue.mpg.de.

based on github.com/akanazawa/hmr
"""
# Sets default args
# Note all data format is NHWC because slim resnet wants NHWC.
import sys
from absl import flags

PRETRAINED_MODEL = './model/ring_6_68641'

flags.DEFINE_string('img_path', '/ps/project/face2d3d/face2mesh/website_release_testings/single_image_test/000001.jpg', 'Image to run')
flags.DEFINE_string('out_folder', './RingNet_output',
                     'The output path to store images')

flags.DEFINE_boolean('save_obj_file', False,
                     'If true the output meshes will be saved')

flags.DEFINE_boolean('save_flame_parameters', False,
                     'If true the camera and flame parameters will be saved')

flags.DEFINE_boolean('neutralize_expression', False,
                     'If true the camera and flame parameters will be saved')

flags.DEFINE_boolean('save_texture', False,
                     'If true the texture map will be stored')

flags.DEFINE_string('flame_model_path', './flame_model/generic_model.pkl', 'path to the neutral FLAME model')

flags.DEFINE_string('flame_texture_data_path', './flame_model/texture_data_512.npy', 'path to the FLAME texture data')


flags.DEFINE_string('load_path', PRETRAINED_MODEL, 'path to trained model')

flags.DEFINE_integer('batch_size', 1,
                     'Fixed to 1 for inference')

# Don't change if testing:
flags.DEFINE_integer('img_size', 224,
                     'Input image size to the network after preprocessing')
flags.DEFINE_string('data_format', 'NHWC', 'Data format')

# Flame parameters:
flags.DEFINE_integer('pose_params', 6,
                     'number of flame pose parameters')
flags.DEFINE_integer('shape_params', 100,
                     'number of flame shape parameters')
flags.DEFINE_integer('expression_params', 50,
                     'number of flame expression parameters')

def get_config():
    config = flags.FLAGS
    config(sys.argv)
    return config
