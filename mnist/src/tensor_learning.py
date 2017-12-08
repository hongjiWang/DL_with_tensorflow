import tensorflow as tf
constant = tf.constant([1, 2, 3])
tensor = constant * constant
sess = tf.Session()
with sess.as_default():
    print(tensor.eval())
