# %%
import tensorflow as tf

a = tf.Variable(5, name='my_variable_a')

a = a.assign_add(a)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

print(sess.run(a))
print(sess.run(a))
# %%
