# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 10:12:27 2020

@author: heyru
"""
import tensorflow as tf
a = tf.constant([10])
b= tf.constant([20])
c=tf.add(a,b)
logs_dir='./logs'
with tf.Session() as sess:
    writer = tf.summary.FileWriter(logs_dir, sess.graph)
    result = sess.run(c)
    print('outcome: ', result)
writer.close()