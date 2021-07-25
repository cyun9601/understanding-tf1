# %%
import tensorflow as tf

with tf.name_scope('scope_alpha'):
    a = tf.constant(5, dtype=tf.int32, name='const_a')
    b = tf.constant(10, dtype=tf.int32, name='const_b')
    c = tf.add(a, b, name='add_c')

with tf.name_scope('scope_beta'):
    d = tf.constant(500, dtype=tf.int32, name='const_a')
    e = tf.constant(1000, dtype=tf.int32, name='const_b')
    f = tf.add(d, e, name='add_c')

output = tf.add(c, f)

sess = tf.Session()

writer = tf.summary.FileWriter('./mygraph', graph=tf.get_default_graph())

print(sess.run(output))
# %%
