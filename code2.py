# %%
import tensorflow as tf 
import numpy as np
# %%
D0_scalar = 15
D1_Vector = [1.5, 2.5, 3.0]
D2_Matrix = [['read a', 'book'], ['write a','book']]
D3_Tensor = [[[True, True], [False, False]], [[True, False], [False, True]]]
# %%
at = tf.constant(D0_scalar, name='input_d0t', dtype=tf.int64)
bt = tf.constant(D1_Vector, name='input_d1t', dtype=tf.float32)
ct = tf.constant(D2_Matrix, name='input_d2t')
dt = tf.constant(D3_Tensor, name='input_d3t', dtype=tf.bool)
# %%
an = tf.constant(np.array(D0_scalar, dtype=np.int64), name='input_d0t')
bn = tf.constant(np.array(D1_Vector, dtype=np.float32), name='input_d1t')
cn = tf.constant(np.array(D2_Matrix), name='input_d2t')
dn = tf.constant(np.array(D3_Tensor, dtype=np.bool), name='input_d3t')

# %%
print(at)
print(an)
print(bt)
print(bn)
print(ct)
print(cn)
print(dt)
print(dn)
# %%

