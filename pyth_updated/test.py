#!/usr/bin/env python
# encoding: utf-8

from datetime import datetime
import os
import random

import tensorflow as tf
import numpy as np

import cpm
import read_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Config():

    # == modify parameters ===========================
    TAG = "_demo" # used for uniform filename
                 # "_demo": train with demo images
               # "": empty) train with ~60000 images
    steps = "516"
    # ================================================


    script_path = os.path.abspath(__file__)
    script_dir = os.path.split(script_path)[0]
    script_base_dir = os.path.split(script_dir)[0]
    script_base_dir = os.path.split(script_base_dir)[0]
    script_base_dir = os.path.split(script_base_dir)[0]


    rel_path_depth_map_file      = "Regression/data/ITOP_top_test_depth_map.h5"
    depth_map_file               = os.path.join( script_base_dir, rel_path_depth_map_file )
    rel_path_label_file          = "Regression/data/ITOP_top_test_labels.h5"
    label_file                   = os.path.join( script_base_dir, rel_path_label_file )
    #rel_model_path               = "%s%s_%s_%s_%s"%("model/",IMAGE_COUNT,FEATURES_COUNT,"regressor","model.pkl")
    #model_path                   = os.path.join( script_base_dir, rel_model_path )

    #print "The path of the testing database is :"
    #print depth_map_file

    #annos_path = "./labels/txt/input/test_annos" + TAG + ".txt"
    #data_path = "./data/input/test_imgs" + TAG
    annos_write_path = "./labels/txt/output/test_annos" + TAG + ".txt"

    image_count = 5
    test_num = 5
    batch_size = 5
    initialize = False
    gpu = '/gpu:0'






    # image config
    points_num = 8
    joint_count_itop = 8
    fm_channel = points_num + 1
    origin_height = 212
    origin_width = 256
    img_height = 216
    img_width = 256
    is_color = False

    # feature map config
    fm_width = img_width >> 1
    fm_height = img_height >> 1
    sigma = 2.0
    alpha = 1.0
    radius = 12

    # random distortion
    degree = 15

    # solver config
    wd = 5e-4
    stddev = 5e-2
    use_fp16 = False
    moving_average_decay = 0.999

    # checkpoint path and filename
    logdir = "./log/test_log/" # no use in test phase
    params_dir = "./params/" + TAG + "/" # no use in test phase

    load_filename =  "cpm" + '-' + steps
    save_filename = "cpm"

def main():
    config = Config()
    with tf.Graph().as_default():

        # create a reader object
        reader = read_data.PoseReader(config)

        # create a model object
        model = cpm.CPM(config)

        # feedforward
        predicts = model.build_fc(False) # False: is not training

        # Initializing operation
        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep=100)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            sess.run(init_op)
            model.restore(sess, saver, config.load_filename)

            # write the predict coordinates to file
            with open(config.annos_write_path, 'w') as fw:
                # start testing
                new_start = 0
                for idx in xrange(config.test_num):
                    with tf.device("/cpu:0"):
                        imgs, fm, coords, begins, filename_list,end_index = \
                        reader.get_random_batch(new_start,distort=False)
                        new_start = end_index + 1
                        # print "imgs"
                        # print np.shape(imgs)
                        # print np.shape(fm)
                        # print np.shape(coords)
                        # print np.shape(begins)
                        # print np.shape(filename_list)
                    # # feed data into the model
                    feed_dict = {
                        model.images: imgs,
                        model.coords: coords,
                        model.labels: fm
                        }
                    with tf.device('/cpu:0'):
                        # run the testing operation
                        predict_coords_list = sess.run(predicts, feed_dict=feed_dict)
                        # print predict_coords_list
                        for filename, predict_coords in zip(filename_list, predict_coords_list):
                            predict_coords = predict_coords.astype(int)
                            print filename
                            print "is the file name"
                            print predict_coords
                            print "are the predicted coordinates"
                            act_coords = reader.read_from_label_dataset(filename)
                            print act_coords
                            print "are the actual coordinates"
                            print "#############################################"
                            #print (filename, predict_coords)
                            #fw.write(filename + ' ')
                            for i in xrange(config.points_num):
                                # w = predict_coords[i*2] * config.img_width
                                # h = predict_coords[i*2 + 1] * config.img_height
                                # item = str(int(w)) + ',' + str(int(h))
                                w = predict_coords[i*2]
                                h = predict_coords[i*2 + 1]
                                item = str(w) + ',' + str(h)
                                fw.write(item + ',')
                            fw.write('1\n')


if __name__ == "__main__":
    print ("Begin testing...")
    main()
    print ("Finished testing.")
