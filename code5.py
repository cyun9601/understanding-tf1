# %%
import tensorflow as tf

a = tf.placeholder(tf.int32, shape=[])
b = tf.constant(7, name='input_b')

c = tf.multiply(a, b, name='mul_c')
d = tf.add(a, b, name='add_d')
e = tf.add(c, d, name='add_e')

sess = tf.Session()
print(sess.run(e, feed_dict={a:10}))
