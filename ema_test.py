import tensorflow as tf


x = tf.Variable(10.0,'float32')

ema_obj = tf.train.ExponentialMovingAverage(0.5)

ema_obj.apply([x])

print(x)

x.assign(x + 10.0)

print(x)

ema_obj.apply([x])

print(x)

ema_obj.average(x))

print(x)