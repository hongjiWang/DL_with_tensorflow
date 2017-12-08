import tensorflow as tf

# create some variable
v1 = tf.get_variable('v1', shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable('v2', shape=[5], initializer=tf.zeros_initializer)

inc_v1 = v1.assign(v1 + 1)
dec_v2 = v2.assign(v2 - 1)

# create the op to initial variables
init_variable = tf.global_variables_initializer()
# create a saver to save the model
saver = tf.train.Saver()

# later, launch the model, initialize the variable, do some work, and save
# the variables to disk
with tf.Session() as sess:
    # initialize the variables
    sess.run(init_variable)
    # do some work with the model
    inc_v1.op.run()
    dec_v2.op.run()
    # save the variables to disk
    save_path = saver.save(sess, "./tmp/learning_save.chpt")
    print("Model saved in file: %s" % save_path)

    # restore the variables from disk
    saver.restore(sess, "./tmp/learning_save.chpt")
    # check the value of the variables
    print("v1: %s" % v1.eval())
    print("v2: %s" % v2.eval())
