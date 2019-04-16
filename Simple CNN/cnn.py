import numpy as np
import tensorflow as tf
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import re
import glob, os

# visualize the images
def display_images(images, labels):
    plt.figure(figsize=(10,10))
    grid_size = min(25, len(images))
    for i in range(grid_size):
        plt.subplot(5, 5,i+1)
        plt.yticks([])
        plt.xticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])


class ImagePrep:
    # initialize all Variables used
    def __init__(self):
        self.maxsize = None
        self.img = None

    # convert any image into grey 8 bit array scale image for preprocessing
    def convert_to_8_bit_grey(self, path, maxsize):
        img = Image.open(path).convert('L') # convert to grey
        # now crop the data to ratio of 1:1
        # cropping works in this data ONLY!
        # as you normaly need to locate the
        # object and crop around it
        WIDTH, HEIGHT = img.size
        if WIDTH != HEIGHT:
            img_size = min(WIDTH, HEIGHT)
            img = img.crop((0, 0, img_size, img_size))
        # scale the image down to the maxsize
        img.thumbnail(maxsize, PIL.Image.ANTIALIAS)
        return np.asarray(img)


        # load the images
    def load_imgs(self, path_dir, maxsize):
        # store the images asarray and labels
        images = []
        labels = []
        os.chdir(path_dir)
        for image in  glob.glob('*.jpg'):
            img = self.convert_to_8_bit_grey(image, maxsize)
            if re.match('chihuahua.*', image):
                images.append(img)
                labels.append(0)
            elif re.match('muffin.*', image):
                images.append(img)
                labels.append(1)
        return (np.asarray(images), np.asarray(labels))


maxsize = 100,100
print(maxsize)
img_prep = ImagePrep()
(train_images, train_labels) = img_prep.load_imgs(os.getcwd(), maxsize)
# reshape the picture dim into 4 ranks
# and reshape train_labels to 16,1
train_images = train_images.reshape(-1,100,100,1)
train_labels = train_labels.reshape(-1,1)

class_names = ['chihuahua', 'muffin']

print(train_images.shape)
print(train_labels)

# show the train images
# display_images(train_images, train_labels)
# plt.show()

# scale down the images
# https://stats.stackexchange.com/questions/185853/why-do-we-need-to-normalize-the-images-before-we-put-them-into-cnn
train_images = train_images/255
train_images = train_images.astype('float32')

im = train_images[:2]
iml = train_labels[:2]
print(im, iml)
# train_images = train_images.all()
# print(next((idx for idx, val in np.ndenumerate(train_images) if val==im)))

# paddings = tf.constant([[2,2],[2,2]])
# tf.pad(im, paddings, "CONSTANT")
# print(im.shape)

# hyperparameters
hidden_nodes  = 1024
learning_rate = 0.01
batch_size    = 2
iters         = 100



# machine learning part
graph = tf.Graph()
with graph.as_default():
    # define place holders for X and y
    X = tf.placeholder(tf.float32, [None, 100, 100,1])
    y_true = tf.placeholder(tf.float32, [None, 1])
    hold_prob = tf.placeholder(tf.float32) # drop out placeholder to decrease overfitting

    # initialize weights
    def weights_init(shape):
        init_w = tf.truncated_normal(shape, 1, -1)
        return tf.Variable(init_w)

    # initialize biases
    def biases_init(shape):
        init_b = tf.Variable(0.1, shape)
        return init_b

    # build conv layer
    def conv2d(x, w, strides):
        return tf.nn.conv2d(x,w, strides=strides,
            padding="VALID"
        )

    # build max pooling layer
    def max_pool(x, shape, strides):
        return tf.nn.max_pool(x, ksize=shape,
            strides=strides, padding="VALID"
        )

    # work the convlution layer
    def convolution(input_x, shape, strides):
        w = weights_init(shape)
        b = biases_init(shape[3])
        return tf.add(conv2d(input_x, w, strides), b)

    # fully connnected layer
    def full_connected(input_layer, hidden_nodes):
        input_size = int(input_layer.get_shape()[1])
        w = weights_init((input_size, hidden_nodes))
        b = biases_init(hidden_nodes)
        return tf.add(tf.matmul(input_layer, w), b)


    # add padding before going to the resnet
    # paddings = tf.constant([[2,2],[2,2]])
    # tf.pad(train_images[0], paddings, "CONSTANT")
    # print(train_images[0].shape)

    # start the network
    # create the layers
    # create the convolutional layer
    # [4,4]

    # 9x9 filter with stride of one (overlapping)
    # this will output 92x92, 32
    convo_1 = tf.nn.relu(convolution(X, [9,9,1,32],[1,1,1,1]))
    print(convo_1.shape)
    # maxpooling 4x4 for the 32 depth of the activation map
    # output 23x23, 32
    convo_1_pooling = max_pool(convo_1,[1,4,4,1],[1,4,4,1])

    # 4x4 filter with stride of 1
    # output 20x20, 64 activation map
    convo_2 = tf.nn.relu(convolution(convo_1_pooling, [4,4,32,64],[1,1,1,1]))
    print(convo_2.shape)
    # maxpooling 4x4 for the depth 32 of the activation map
    # output 5x5, 64*32
    convo_2_pooling = max_pool(convo_2, [1,4,4,1],[1,4,4,1])
    print(convo_2_pooling.shape)

    #create flattening
    convo_2_flat = tf.reshape(convo_2_pooling, [-1, 5*5*64])
    # create the layers
    full_layer_one = tf.nn.relu(full_connected(convo_2_flat, hidden_nodes))
    full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)
    y_pred = full_connected(full_one_dropout, 1)


    # create the loss function
    # we are using cross entropy
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))

    # create the optimizer
    # we are using adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cross_entropy)


# start the training session
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    # the iteration for the training
    print("Start training")
    for i in range(iters):
        sess.run(train, feed_dict={X: train_images, y_true: train_labels, hold_prob: 0.5})



        # record the trianing
        if i % iters == 0:

            print(f"Currently at step {i}")
            print("Accuracy is: ")

            # test the train nodes
            acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(y_true, 1),
                                  predictions=tf.argmax(y_pred,1))
            print(f"{acc}")
            print("\n")
