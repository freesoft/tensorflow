import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# adding 22 actual sqft, price, and bedroom(s) of the house sold recently, train them, and estimate three new house price based on sqft when it's on the market

num_points = 22 
vectors_set = []
# x : sqft, y: # of bedrooms, z : price
vectors_set.append([1250, 2, 580000])
vectors_set.append([1230, 2, 540000])
vectors_set.append([1300, 3, 600000])
vectors_set.append([1310, 3, 625000])
vectors_set.append([1400, 2, 620000])
vectors_set.append([1450, 3, 625000])
vectors_set.append([1420, 3, 625000])
vectors_set.append([1600, 3, 670000])
vectors_set.append([1630, 3, 650000])
vectors_set.append([1610, 3, 682000])
vectors_set.append([1750, 3, 700000])
vectors_set.append([1880, 4, 720000])
vectors_set.append([1900, 4, 730000])
vectors_set.append([2100, 3, 750000])
vectors_set.append([2200, 4, 790000])
vectors_set.append([2250, 4, 770000])
vectors_set.append([2300, 4, 790000])
vectors_set.append([1230, 2, 574000])
vectors_set.append([1680, 2, 790000])
vectors_set.append([1815, 3, 720000])
vectors_set.append([2020, 3, 770000])
vectors_set.append([2400, 4, 810000])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]
z_data = [v[2] for v in vectors_set]

W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
z = W1 * x_data + W2 * y_data + b

loss = tf.reduce_mean(tf.square(z - z_data))

optimizer = tf.train.GradientDescentOptimizer(0.00000005)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in xrange(15):
     sess.run(train)
     print(step, sess.run(W1), sess.run(b))
     print(step, sess.run(W2), sess.run(b))
     print(step, sess.run(loss))

# visualize
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = meshgrid(x_data, y_data)
surf = ax.plot_surface(X, Y, z.eval(session=sess), rstride=10, cstride=1, cmap=cm.RdBu,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
# Set rotation angle to 90 degrees since there isn't much visibility with default angle.
ax.view_init(azim=90)
plt.show()

    
# now we get W1, W2, b, so we can guess how much house price is going to be for given sqft and # of bedrooms
estimate_vectors_set = []
estimate_vectors_set.append([1250,2])
estimate_vectors_set.append([1780,3])
estimate_vectors_set.append([1850,3])
estimate_vectors_set.append([2300,4])

est_x_data = [v[0] for v in estimate_vectors_set]
est_y_data = [v[1] for v in estimate_vectors_set]
est_z_data = [v[0] * sess.run(W1) + v[1] * sess.run(W2) + sess.run(b)  for v in estimate_vectors_set]

# print estimate sqft/price 
for i  in range(len(est_x_data)):
	print(est_x_data[i], est_y_data[i], est_z_data[i]) 


# Result will be something like this, 1250 sqft 2 bedroom will be $488,548 USD, etc
#(1250, 2, array([ 488548.09375], dtype=float32))
#(1780, 3, array([ 695692.375], dtype=float32))
#(1850, 3, array([ 723051.0625], dtype=float32))
#(2300, 4, array([ 898928.3125], dtype=float32))
