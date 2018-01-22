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

  # =================== modify parameters ==================
  TAG = "_demo" # used for uniform filename
               # "_demo": train with demo images
               # "": (empty) train with ~60000 images
               # "ITOP" for ITOP dataset
  batch_size = 20
  initialize = False # True: train from scratch (should also
                    # delete the corresponding params files
                    # in params_dir);
                     # False: restore from pretrained model
  steps = "100000"   # if 'initialize = False', set steps to
                     # where you want to restore
  toDistort = False
  image_count = 40000
  # iterations config
  max_iteration =8
  checkpoint_iters = 2
  #no_epoch = (int)( image_count / batch_size )
  summary_iters = 2
  validate_iters = 2


  # ========================================================

  #annos_path = "./labels/txt/input/train_annos" + TAG + ".txt"
  data_path = "./data/input/train_imgs" + TAG + "/"
  gpu = '/gpu:0'

  # checkpoint path and filename
  logdir = "./log/train_log/"
  params_dir = "./params/" + TAG + "/"
  #load_filename = "cpm" + '-' + steps
  load_filename =  "cpm" + '-' + steps
  save_filename = "cpm"
 # ========================================================
 # parameters of itop database

  script_path = os.path.abspath(__file__)
  script_dir = os.path.split(script_path)[0]
  script_base_dir = os.path.split(script_dir)[0]
  script_base_dir = os.path.split(script_base_dir)[0]
#  script_base_dir = os.path.split(script_base_dir)[0]

  rel_path_depth_map_file      = "RF/data/ITOP_top_train_depth_map.h5"
  depth_map_file               = os.path.join( script_base_dir, rel_path_depth_map_file )
  rel_path_label_file          = "RF/data/ITOP_top_train_labels.h5"
  label_file                   = os.path.join( script_base_dir, rel_path_label_file )
  #rel_model_path               = "%s%s_%s_%s_%s"%("model/",IMAGE_COUNT,FEATURES_COUNT,"regressor","model.pkl")
  #model_path                   = os.path.join( script_base_dir, rel_model_path )

  #print "The path of the training database is :"
  #print depth_map_file

 # ========================================================

  # image config
  points_num = 8
  joint_count_itop = 8
  fm_channel = points_num + 1
  origin_height = 216
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
  degree = 8

  # solver config
  wd = 5e-4
  stddev = 5e-2
  use_fp16 = False
  moving_average_decay = 0.999

def main():

    config = Config()
    with tf.Graph().as_default():
        # create a reader object
        reader = read_data.PoseReader(config)

        # create a model object
        model = cpm.CPM(config)

        # feedforward
        predicts = model.build_fc(True)

        # return the loss
        loss = model.loss()

        # training operation
        train_op = model.train_op(loss, model.global_step)
        # Initializing operation
        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep = 100)

        sess_config = tf.ConfigProto(log_device_placement=True)
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:

            # initialize parameters or restore from previous model
            if not os.path.exists(config.params_dir):
                os.makedirs(config.params_dir)
            if os.listdir(config.params_dir) == [] or config.initialize:
                print ("Initializing Network")
                sess.run(init_op)
            else:
                sess.run(init_op)
                model.restore(sess, saver, config.load_filename)

            merged = tf.summary.merge_all()
            logdir = os.path.join(config.logdir,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

            writer = tf.summary.FileWriter(logdir, sess.graph)
            new_start = 0
            offset = 0
            epoch_count = 1
            print "Last ", config.image_count % config.batch_size, "will be missed in every ", config.image_count , "images"
            # start training
            for idx in xrange(config.max_iteration):
                with tf.device(config.gpu):
                  if ( offset + config.image_count) <= (idx * config.batch_size):
                  #    new_start = 0
                      offset = ((idx-1) * config.batch_size)
                      epoch_count = epoch_count + 1
                      print "start of new epoch"
                  imgs, fm, coords, begins, filename_list,end_index = \
                  reader.get_random_batch( new_start, distort=config.toDistort )
                  new_start = end_index + 1
                  print filename_list
                  print "are the file names"
                  print coords
                  print "are the coordinates"
                # feed data into the model
                feed_dict = {
                    model.images: imgs,
                    model.coords: coords,
                    model.labels: fm
                    }
                with tf.device(config.gpu):
                    # run the training operation
                    sess.run(train_op, feed_dict=feed_dict)

                with tf.device(config.gpu):
                  # write summary
                  if (idx + 1) % config.summary_iters == 0:
                      print ("iter#", idx + 1)
                      tmp_global_step = model.global_step.eval()
                      summary = sess.run(merged, feed_dict=feed_dict)
                      writer.add_summary(summary, tmp_global_step)
                  # save checkpoint
                  if (idx + 1) % config.checkpoint_iters == 0:
                      tmp_global_step = model.global_step.eval()
                      save_path = model.save(sess, saver, config.save_filename, tmp_global_step)



if __name__ == "__main__":
    main()
