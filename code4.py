# %%
import tensorflow as tf

g1 = tf.get_default_graph()
g2 = tf.Graph()

with g1.as_default():
    a1 = tf.constant(5, name='input_a1')
    b1 = tf.constant(7, name='input_b1')

    c1 = tf.multiply(a1, b1, name='mul_c1')
    d1 = tf.add(a1, b1, name='add_d1')
    e1 = tf.add(c1, d1, name='add_e1')

with g2.as_default():
    a2 = tf.constant(5, name='input_a2')
    b2 = tf.constant(7, name='input_b2')

    c2 = tf.multiply(a2, b2, name='mul_c2')
    d2 = tf.add(a2, b2, name='add_d2')
    e2 = tf.add(c2, d2, name='add_e2')

sess1 = tf.Session()
sess2 = tf.Session(graph = g2)

print(sess1.run(e1))
print(sess2.run(e2))

# %%
writer1 = tf.summary.FileWriter('./mygraph', sess1.graph)
# writer2 = tf.summary.FileWriter('./mygraph', sess2.graph)

# %%
