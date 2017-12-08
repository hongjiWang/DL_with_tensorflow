import tensorflow as tf

k = tf.placeholder(tf.float32)
#
# # make a normal distribution, with shiftting mean
# mean_moving_normal = tf.random_normal(shape=[1000], mean=(5 * k), stddev=1)
#
# # record that distribution in a histogram summary
# tf.summary.histogram("normal/moving_mean", mean_moving_normal)
#
# # setup a session and summary writer
# sess = tf.Session()
# writer = tf.summary.FileWriter("tmp/histogram_example")
#
# summaries = tf.summary.merge_all()
#
# # setup a loop and write the summaries to disk
# N = 400
# for step in range(N):
#     k_val = step/float(N)
#     summ = sess.run(summaries, feed_dict={k: k_val})
#     writer.add_summary(summ,global_step=step)
#

mean_moving_normal = tf.random_normal(shape=[1000], mean= (5 * k), stddev=1)
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

variable_shrinking_normal = tf.random_normal(shape=[1000], mean=0, stddev=1 - (k))
tf.summary.histogram("normal/shrinking_variable", variable_shrinking_normal)

normal_combined = tf.concat([mean_moving_normal, variable_shrinking_normal], 0)
tf.summary.histogram("normal/bimodal", normal_combined)

summaries = tf.summary.merge_all()
sess = tf.Session()
writer = tf.summary.FileWriter("/tmp/histogram_example")

N = 400
for step in range(N):
    k_val = step/float(N)
    summ = sess.run(summaries, feed_dict={k: k_val})
    writer.add_summary(summ, global_step=step)

    