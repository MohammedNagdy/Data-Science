# data from https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/  download wikitext-103
import numpy as np
import tensorflow as tf
import datetime
import random

# load the text file
f = open("wikitext-103/wiki.test.tokens").read()
print(f"the length of the test text: {len(f)}")
# print(f"""head of the text \n \
#         {f[:1000]}""")

# sort the characters and print them out
chars = sorted(list(set(f)))
char_size = len(chars)
print(f" the size of chars: {char_size}")
# print(chars)


# convert characters to ids
id3char = dict((i, c) for i, c in enumerate(chars))
char2id = dict((c, i) for i, c in enumerate(chars))

# generate probabilty for each next character
def sample(prediction):
    r = random.uniform(0,1)
    # store the prediction character
    s = 0
    char_id = len(prediction) -1
    # iterate through the length of each char
    for i in range(len(prediction)):
        s += prediction[i]
        if s >= r:
            char_id = i
            break

    # one hot encoding part
    char_one_hot = np.zeros(shape(char_size))
    char_one_hot[char_id] = 1.0
    return char_one_hot

# vectorize our data to feed into the model
# length of the sentence
len_per_section = 50
# skip the characters that were predicted
skip = 2 # using small set of characters to skip, will overlap with the previous characters and make the madel less accurate
sections = []
next_chars = []

for i in range(0, len(f) - len_per_section, skip):
    sections.append(f[i: i + len_per_section])
    next_chars.append(f[i + len_per_section])

print(len(sections))

# vectorization part ( turning words into vectors)
# matrix of the section length as features
X = np.zeros((len(sections), len_per_section, char_size))
# matrix of the label
y = np.zeros((len(sections), char_size))
# convert both the features and the label for each character in them
# to its corresponding id
for i, section in enumerate(sections):
    for j, char in enumerate(section):
        X[i, j, char2id[char]] = 1
    y[i , char2id[next_chars[i]]] = 1


# deep learning part
# hyperparameters
batch_size = 512
n_iter = 70000
log = 100
save_every = 10000
hidden_nodes = 1024
# starting text ot genrate from
test_start = "I am king"
# create a checkpoint to see where we are in the training process
checkpoint_directory = "ckpt"

# creating a checkpoint
if tf.gfile.Exists(checkpoint_directory):
    tf.gfile.DeleteRecursively(checkpoint_directory)
tf.gfile.MakeDirs(checkpoint_directory)

print(f"training data size: {len(X)}")
print(f"approximate iterations per epoch: {int(len(X)/batch_size)}")

# build the model
graph = tf.Graph()
# to define which graph we're using if we have multiple graphs
with graph.as_default():
    # it shows us where wa are in the iterations
    global_iter = tf.Variable(0)
    # define placeholdeers
    data = tf.placeholder(tf.float32, (batch_size, len_per_section, char_size))
    labels = tf.placeholder(tf.float32, (batch_size, char_size))

    # define our gates in the lstm (input, output, forget, hidden state)
    # these will calculated in parallel not related to each other

    # input gate matrices
    # the weights of the inputs and bias
    w_ii = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
    w_io = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_i = tf.Variable(tf.zeros([1, hidden_nodes]))

    # forget gate matrices
    w_fi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
    w_fo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_f = tf.Variable(tf.zeros([1, hidden_nodes]))

    # output gate matrices
    w_oi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
    w_oo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_o = tf.Variable(tf.zeros([1, hidden_nodes]))

    # hidden cell state gate
    w_ci = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
    w_co = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_c = tf.Variable(tf.zeros([1, hidden_nodes]))

    # create the lstm cell
    def lstm(i, o, state):
        # these are calculated sperately no overlap
        # (input * input_weights) + (output * weights for previous output gate) + bias
        input_gate = tf.sigmoid(tf.matmul(i, w_ii) + tf.matmul(o, w_io) + b_i)

        # (input * forget_weights) + (output * weights for the previous output) + bias
        forget_gate = tf.sigmoid(tf.matmul(i, w_fi) + tf.matmul(o, w_fo) + b_f)

        # (input * output_weights) + (output * weights for previous output) + bias
        output_gate = tf.sigmoid(tf.matmul(i, w_oi) + tf.matmul(o, w_oo) + b_o)

        # (input * hidden cell state weights) + (output * wieghts for the previous output) + bias
        memory_cell = tf.sigmoid(tf.matmul(i, w_ci) + tf.matmul(o, w_co) + b_c)

        # to update get our new cell state in this lstm cell
        # cell state = forget gate * given sate + input gate * memory_cell
        state = forget_gate * state + input_gate * memory_cell

        # squash the state with hyperbolic tangent ( compute hyperbolic tangent of x element wise) then multiply by output
        output = output_gate * tf.tanh(state)

        return output, state


    # operations for the lstm
    # start with empty, lstm will calculate this
    output = tf.zeros([batch_size, hidden_nodes])
    state = tf.zeros([batch_size, hidden_nodes])

    # unrolled LSTM loop
    # for each input set
    for i in range(len_per_section):
        # calc the stae and output from LSTM
        output, state = lstm(data[:, i, :], output, state)
        # in the start
        if i == 0:
            # store initial output and labels
            outputs_all_i = output
            labels_all_i = data[:, i+1 , :]

        # for each set concat the labels and the outputs
        elif i != (len_per_section -1) :
            outputs_all_i = tf.concat([outputs_all_i, output], 0)
            labels_all_i = tf.concat([labels_all_i, data[:, i+1, :]], 0)

        # for the last character
        else:
            # final store
            outputs_all_i = tf.concat([outputs_all_i, output], 0)
            labels_all_i = tf.concat([labels_all_i, labels], 0)



        # classifier
        # classifier will a run after the saved output and saved state were assigned

        # calc the weights and bias for the neuorns
        # the weights are genrated randomnly using a specific distibution
        w = tf.Variable(tf.truncated_normal([hidden_nodes, char_size], -0.1,0.1))
        b = tf.Variable(tf.zeros([char_size]))

        # use gpu
        with tf.device('/gpu:0'):
            # now we'll define logits so we can use it in the loss function
            logits = tf.matmul(outputs_all_i, w) + b

            # define the loss function cross entropy (for multi class classifications)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_all_i))

            # optimizer
            optimizer = tf.train.GradientDescentOptimizer(10).minimize(loss, global_step=global_iter)

# start the session for training
with tf.Session(graph=graph) as sess:
    # standard init step
    tf.global_variables_initializer().run()
    offset = 0
    saver = tf.train.Saver()

    # for each training step
    for step in range(n_iter):

        # starts with offset as 0
        offset = offset % len(X)

        # calc batch data and labels to feed the model iteratively
        if offset <= (len(X) -batch_size):
            # first part
            batch_data = X[offset: offset + batch_size ]
            batch_labels = y[offset: offset + batch_size ]
            offset += batch_size

        # until when offset = batch size then we
        else:
            to_add = batch_size- (len(X) - offset)
            batch_data = np.concatenate((X[offset:len(X)], X[0: to_add]))
            batch_labels = np.concatenate((y[offset: len(X)], y[0, to_add]))
            offset = to_add

        # OPTIMIZE!!!
        _, training_loss = sess.run([optimizer, loss], feed_dict={data: batch_data, labels : batch_labels})

        if step % 10 == 0:
            print(f"training loss at step {step} is {training_loss} at time {datetime.datetime.now()}")

        if step % save_every == 0:
            saver.save(sess, checkpoint_directory+'/model', global_step=step)



# the predicting part
test_start = "I love my mom"

with tf.Session(graph=graph) as sess:
    #init graph load model
    tf.global_variables_initializer().run()
    model = tf.train.latest_checkpoint(checkpoint_directory)
    saver = tf.train.Saver()
    saver.restore(sess, model)

    # set input variables to generate chars from
    reset_test_state.run()
    test_generated = test_start

    # for every character in input sentence
    for i in range(len(test_start) - 1):
        # initialize an empty char store
        test_X = np.zeros((1, char_size))
        # store it in id form
        test_X[0, char2id[test_start[i]]] = 1.
        # feed it to model, test prediction is output value
        _ = sess.run(test_preciction, feed_dict={test_data: test_X})


    # where we store character encoded prediction
    test_X = np.zeros((1, char_size))
    test_X[0, char2id[test_start[-1]]] = 1.

    # generating characters
    for i in range(500):
        # get each prediction probabilty
        prediction = test_prediction.eval({test_data: test_X})[0]

        # one hot encode it
        next_char_one_hot = sample(prediction)

        # get the indices of the max values (highest probabilty) and convert to char
        next_char = id2char[np.argmax(next_char_one_hot)]

        # add each character to output test to output iteratively
        test_generated += next_char
        # update
        test_X = next_char_one_hot.reshape((1, char_size))

    print(text_generated)
