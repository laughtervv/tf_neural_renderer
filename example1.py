import tensorflow as tf
import tf_neuralrenderer

import numpy as np
import scipy.misc
import tqdm
import neural_renderer

import argparse
import glob
import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
working_directory = os.path.dirname('result/')

with tf.Session('', config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # with tf.device('/cpu:0'):

    # Official Test1: test render with mask
    vertices, faces = neural_renderer.load_obj('./data/teapot.obj')
       
    renderer = neural_renderer.Renderer()
    nmr = tf_neuralrenderer.NMR(renderer)

    verts = tf.constant(np.expand_dims(vertices, axis=0))
    tris = tf.constant(np.expand_dims(faces, axis=0))

    # other settings
    camera_distance = 2.732
    elevation = 30
    texture_size = 2
    textures = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
    print (verts.shape, tris.shape)
    print(np.asarray([1, vertices.shape[0]], dtype=np.int), np.asarray([[faces.shape[0]]], dtype=np.int))
    img = nmr.neural_renderer_texture(verts, np.asarray([[ vertices.shape[0]]], dtype=np.int), 
    								  tris, np.asarray([[faces.shape[0]]], dtype=np.int), 
    								  textures, 'render')

    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        nmr.renderer.eye = neural_renderer.get_points_from_angles(camera_distance, elevation, azimuth)
        img_val = sess.run([img])
        print(img_val[0][0].shape)
        # images = nmr.renderer.render(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
        image = img_val[0][0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
        scipy.misc.toimage(image, cmin=0, cmax=1).save('%s/_tmp_%04d.png' % (working_directory, num))

    # generate gif (need ImageMagick)
    options = '-delay 8 -loop 0 -layers optimize'
    subprocess.call('convert %s %s/_tmp_*.png %s' % (options, working_directory, 'result/example1.gif'), shell=True)

    # remove temporary files
    for filename in glob.glob('%s/_tmp_*.png' % working_directory):
        os.remove(filename)
