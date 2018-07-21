import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import glob
import os
import numpy as np
from collections import namedtuple



def simple_example():
  a = tf.constant([1,2,3,4])
  b = tf.constant([3,4,5,6])

  product = tf.multiply(a,b)
  print(product)


  #To run the session do this
  sesh = tf.Session()

  print(sesh.run(product))

  sesh.close()

  # running interactively 
  with tf.Session() as sesh:
      
      print(sesh.run(product))


  # In[5]:


  # passing configuration options
  # Log device placement
  config=tf.ConfigProto(log_device_placement=True)
  # For allowing system to execute on CPU when GPU is not available or instructions cannot be placed on GPU.
  config=tf.ConfigProto(allow_soft_placement=True)
   
  with tf.Session() as sesh:
      print(sesh.run(product))
    
FLAGS = None




def simple_nn():

  sess = tf.InteractiveSession()

  # create the Graph

  inputs = tf.placeholder(dtype = tf.float32, shape = [None,784])
  labels = tf.placeholder(dtype = tf.int32, shape = [None,10])

  W = tf.Variable(tf.zeros([784,10]))
  b = tf.Variable(tf.zeros([10]))

  sess.run(tf.global_variables_initializer())


  y = tf.matmul(inputs,W)+b

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=y))

  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  for _ in range(1000):
      batch = mnist.train.next_batch(100)
      train_step.run(feed_dict={inputs: batch[0],labels: batch[1]})


  correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(labels,1))

  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  print(accuracy.eval(feed_dict= {inputs: mnist.test.images, labels:mnist.test.labels}))

# Data pipeline

def get_dataset(folder_name):
  class_names= glob.glob(folder_name+'/*')
  filenames =[]
  labels = []
  print(len(class_names))
  for idx,item in enumerate(class_names):
    files = glob.glob(os.path.join(folder_name,item)+'/*.jpg')
    filenames.extend(files)
    labels.extend(np.ones((1,len(files)))*idx)
  labels = np.array(labels,np.uint8)
  labels = labels.ravel()
  print(labels.shape)
  filenames=np.array(filenames)

  # Split into training and test
  split_ratio = 0.7
  select_train = np.random.choice(len(filenames),int(len(filenames)*split_ratio))
  select_val = np.setdiff1d(np.arange(len(filenames)),select_train)
  print(select_val)
  training_files = filenames[select_train]
  training_labels = labels[select_train]
  validation_files = filenames[select_val]
  validation_labels = labels[select_val]
  dataset = {'training_files':training_files,'training_labels':training_labels
              ,'validation_files':validation_files,'validation_labels':validation_labels}
  return dataset


NUM_CLASSES=2

def input_parser(img_path, label):
    # convert the label to one-hot encoding
    #one_hot = tf.one_hot(label, NUM_CLASSES)
    one_hot=label
    # read the img from file
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)
    print(tf.rank(img_decoded))
    
    img_decoded=tf.image.resize_image_with_crop_or_pad(img_decoded,28,28)
    return img_decoded, one_hot


#helper functions to generate weights
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# In[ ]:

def conv2d(inputs,W):
  return tf.nn.conv2d(inputs,W,strides=[1,1,1,1],padding='SAME')

def pool2d(inputs):
  return tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


#Performing Convolutionaly

dataset = get_dataset('/Users/midhun/Downloads/kagglecatsanddogs_3367a/PetImages')


tr_data = tf.data.Dataset.from_tensor_slices((dataset['training_files'],dataset['training_labels']))
val_data = tf.data.Dataset.from_tensor_slices((dataset['validation_files'],dataset['validation_labels']))

tr_data = tr_data.map(input_parser)

tr_data = tr_data.repeat()
tr_data = tr_data.batch(10)

tr_data =tr_data.apply(tf.contrib.data.ignore_errors())

val_data= val_data.map(input_parser)



iterator = tf.data.Iterator.from_structure(tr_data.output_types,
                                   tr_data.output_shapes)
#val_iterator = tf.data.Iterator.from_structure(val_data.output_types,
#                                   tr_data.output_shapes)

next_element = iterator.get_next()

#next_validation = iterator.get_next()



training_init_op = iterator.make_initializer(tr_data)
#validation_init_op = iterator.make_initializer(val_data)


# with tf.Session() as sess:

#   sess.run(training_init_op)

#   while(True):
#     try:
#       elem=sess.run(next_element)
#       print(elem[0].shape)
#     except tf.errors.OutOfRangeError:
#       print("End of dataset")
#       break



#mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data")

x = tf.placeholder(dtype = tf.float32, shape = [None,28,28,3])
labels = tf.placeholder(tf.int64,[None])

inputs = tf.reshape(x,[-1,28,28,3])

W_c1 = weight_variable([5,5,3,32])
b_c1 = weight_variable([32])

conv1 = tf.nn.relu(conv2d(inputs,W_c1)+b_c1)

pool1 = pool2d(conv1)

W_c2 = weight_variable([5,5,32,64])
b_c2 = weight_variable([64])

conv2 = tf.nn.relu(conv2d(pool1,W_c2)+b_c2)

pool2 = pool2d(conv2)



flatten2 = tf.reshape(pool2, [-1, 7*7*64])


# FC layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])


h_fc1 = tf.nn.relu(tf.matmul(flatten2,W_fc1)+b_fc1)
keep_prob = tf.placeholder(tf.float32)
out= tf.layers.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv =  tf.matmul(out, W_fc2) + b_fc2

print(y_conv)

cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=y_conv)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
print(y_conv,labels)
correct_prediction =  tf.equal(tf.argmax(y_conv,1),labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(training_init_op)
    for i in range(20000):
      #batch = mnist.train.next_batch(50)
      batch = sess.run(next_element)
      #print(batch)
      if i % 100 == 0:
        loss_value = cross_entropy.eval(feed_dict={x:batch[:][0], labels: batch[:][1],keep_prob: 1.0})
        train_accuracy = accuracy.eval(feed_dict={x:batch[:][0] , labels: batch[:][1], keep_prob: 1.0})
        print('step %d, training accuracy %g , Loss : %.6f' % (i, train_accuracy,loss_value))
      train_step.run(feed_dict={x: batch[:][0], labels: batch[:][1], keep_prob: 0.5})
    #print('test accuracy %g' % accuracy.eval(feed_dict={
     # x: next_validation[:][0], labels:next_validation[:][1] , keep_prob: 1.0}))

