# coding:utf-8
import tensorflow as tf

tf.app.flags.DEFINE_string('file_path','./dataset','ff')
FLAGS=tf.app.flags.FLAGS
print(FLAGS.file_path)
