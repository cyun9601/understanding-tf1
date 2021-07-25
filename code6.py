# %%
import tensorflow as tf

a = tf.Variable(12, name='my_variable')
b = tf.constant(7, name='input_b')

c = tf.multiply(a, b, name='mul_c')
d = tf.add(a, b, name='add_d')
e = tf.add(c, d, name='add_e')

# 변수 사용을 위해 초기화 
init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)
print(sess.run(e))
# %%
