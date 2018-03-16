import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

pred_0_1_tf = tf.linspace(0.0, 1.0, num=100)
pred_0_1 = np.linspace(0.0, 1.0, num=100)
tf.cast(pred_0_1,tf.float32)
dummy_target = 0.0
tf.cast(dummy_target,tf.float32)
y_cross_entropy = []
y_square_error = []

for i in range(0,100):
    ce_loss = -dummy_target * tf.log(pred_0_1_tf[i]) - (1-dummy_target) * tf.log(1-pred_0_1_tf[i])
    y_cross_entropy.append(sess.run(ce_loss))
    se_loss = tf.divide(tf.sqrt(tf.square(pred_0_1_tf[i])),1)
    y_square_error.append(sess.run(se_loss))

line, = plt.plot(pred_0_1, y_cross_entropy, 'r', label="loss function: cross-entropy")
line, = plt.plot(pred_0_1, y_square_error, 'b', label="loss function: squared-error")

plt.legend(loc='upper left', shadow=True, fontsize='small')


plt.ylabel('loss value')
plt.xlabel('y prediction')
plt.show()
