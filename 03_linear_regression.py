import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# adding 22 actual sqft and price of the house sold recently, train them, and estimate three new house price based on sqft when it's on the market

num_points = 22 
vectors_set = []
# x : sqft, y : house price
vectors_set.append([1250, 580000])
vectors_set.append([1230, 540000])
vectors_set.append([1300, 600000])
vectors_set.append([1310, 625000])
vectors_set.append([1400, 620000])
vectors_set.append([1450, 625000])
vectors_set.append([1420, 625000])
vectors_set.append([1600, 670000])
vectors_set.append([1630, 650000])
vectors_set.append([1610, 682000])
vectors_set.append([1750, 700000])
vectors_set.append([1880, 720000])
vectors_set.append([1900, 730000])
vectors_set.append([2100, 750000])
vectors_set.append([2200, 790000])
vectors_set.append([2250, 770000])
vectors_set.append([2300, 790000])
vectors_set.append([1230, 574000])
vectors_set.append([1680, 790000])
vectors_set.append([1815, 720000])
vectors_set.append([2020, 770000])
vectors_set.append([2400, 810000])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

plt.plot(x_data, y_data, 'ro', label='house price')
plt.legend()
plt.show()

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.00000005)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(15):
     sess.run(train)
     print(step, sess.run(W), sess.run(b))
     print(step, sess.run(loss))

     #Graphic display
     plt.plot(x_data, y_data, 'ro')
     plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
     plt.xlabel('sqft')
     plt.ylabel('price(k)')
     plt.legend()
     plt.show()

# now we get W, b, so we can guess how much house price is going to be for given sqft
estimate_vectors_set = []
estimate_vectors_set.append([1250])
estimate_vectors_set.append([1780])
estimate_vectors_set.append([2300])

est_x_data = [v[0] for v in estimate_vectors_set]
est_y_data = [v[0] * sess.run(W) + sess.run(b)  for v in estimate_vectors_set]

# print estimate sqft/price 
for i  in range(len(est_x_data)):
	print(est_x_data[i], est_y_data[i]) 

# draw plot
plt.plot(est_x_data, est_y_data, 'ro', label='estimate')
plt.legend()
plt.show()

