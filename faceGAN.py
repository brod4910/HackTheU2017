import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


# Make a queue of file names including all the JPEG images files in the relative
# image directory.
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("./data/*.jpg"))

print(filename_queue.size())

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()
#
# # Read a whole file from the queue, the first returned value in the tuple is the
# # filename which we are ignoring.
_, image_file = image_reader.read(filename_queue)
#
# # Decode the image as a JPEG file, this will turn it into a Tensor which we can
# # then use in training.
input_data = tf.image.decode_jpeg(image_file, channels=1)

input_data = tf.cast(input_data, tf.float32)

input_data.set_shape([250, 250, 1])

print(input_data)
# 8000 training and 5000 tests

final_data = tf.reshape(input_data, [-1])

print(final_data)

# from keras.preprocessing import image
#
# import matplotlib.pyplot as plt
#
# img_path = "./data/Abdoulaye_Wade_0002.jpg"
# img = image.load_img(img_path, grayscale=True, target_size=(250, 250))
# print(type(img))
#
# x = image.img_to_array(img)
# print(type(x))
# print(x.shape)
#
# newArray = x.flatten()
# print(type(newArray))
# print(newArray.shape)


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, 62500])

D_W1 = tf.Variable(xavier_init([62500, 256]))
D_b1 = tf.Variable(tf.zeros(shape=[256]))

D_W2 = tf.Variable(xavier_init([256, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


Z = tf.placeholder(tf.float32, shape=[None, 100])

G_W1 = tf.Variable(xavier_init([100, 256]))
G_b1 = tf.Variable(tf.zeros(shape=[256]))

G_W2 = tf.Variable(xavier_init([256, 62500]))
G_b2 = tf.Variable(tf.zeros(shape=[62500]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(250, 250), cmap='Greys_r')

    return fig


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 1
Z_dim = 100

# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

sess.run(tf.local_variables_initializer())



# Get an image tensor and print its value.
# image_tensor = sess.run([final_data])
# print(image_tensor)
# npa = np.asarray(image_tensor, dtype=np.float32)
# print(npa.shape)
# print(npa)
# print(type(npa))




if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(10000):
    print(it)
    if it % 500 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb = tf.train.batch([final_data], batch_size=128)

    # print(X_mb)

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    a = sess.run(X_mb)

    # Finish off the filename queue coordinator.
    # coord.request_stop()
    # coord
    # # coord.join(threads)
    # coord.clear_stop()

    # print(a)
    # print(type(a))
    # npa = np.asarray(a, dtype=np.float32)
    # print(npa.shape)
    # print(npa)
    # print(type(npa))
    # print(X_mb.shape)
    #
    # a = X_mb.eval(session=sess)

    # X_mb = npa

    # final_data = np.vstack([1, newArray])
    # print(type(final_data))
    # print(final_data.shape)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: a, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    if it % 500 == 0:
        print('Iter: {}'.format(it))
        print('D_loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()



# session = tf.InteractiveSession()
#
# x = tf.placeholder(tf.float32, shape=[None, 784])  # 62500
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
#
# W = tf.Variable(tf.zeros([784, 10]))  # 62500
#
# b = tf.Variable(tf.zeros([10]))
#
# session.run(tf.global_variables_initializer())
#
# y = tf.matmul(x,W) + b
#
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# for _ in range(1000):
#     batch = mnist.train.next_batch(100)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#
#
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
