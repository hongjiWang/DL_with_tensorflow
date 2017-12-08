import tensorflow as tf
node1 = tf.constant(1.0)
node2 = tf.constant(2.0)
node3 = tf.add(node1, node2)
tf.summary.scalar("tmp/node3", node3)
# sum_graph = tf.Graph()

summaries = tf.summary.merge_all()
sess = tf.Session()
writer = tf.summary.FileWriter('/tmp/sum_a_and_b', sess.graph)
summ = sess.run(summaries)
writer.add_summary(summ)
# writer.add_graph(sum_graph)
print(sess.run(node3))