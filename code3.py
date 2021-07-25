# %%
import tensorflow as tf

g = tf.get_default_graph()

a = tf.constant(5, name='input_a')
b = tf.constant(7, name='input_b')

c = tf.multiply(a, b, name='mul_c')
d = tf.add(a, b, name='add_d')
e = tf.add(c, d, name='add_e')

sess = tf.Session(graph=g)
print(sess.run(e))
# %%
writer = tf.summary.FileWriter('./mygraph', sess.graph)
# %%
