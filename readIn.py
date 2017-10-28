# Typical setup to include TensorFlow.
import tensorflow as tf
import numpy as np

# Make a queue of file names including all the JPEG images files in the relative
# image directory.
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("./data/*.jpg"))

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
_, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_jpeg(image_file, channels=1)

input_data = tf.cast(image, tf.float32)

input_data.set_shape([250, 250, 1])

print(input_data)

final_data = tf.reshape(input_data, [-1])

print(final_data)

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.local_variables_initializer().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    image_tensor = sess.run([final_data])
    print(image_tensor)
    npa = np.asarray(image_tensor, dtype=np.float32)
    print(npa.shape)
    print(npa)
    print(type(npa))

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)


