import os
import gzip
import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import itertools


def load_mnist(path, kind='train'):

    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        
    return images, labels

def generatebatch(X,Y, batch_size, i):
    start = batch_size * i
    if start >= Y.shape[0]:
        start = start % Y.shape[0]
    end = start + batch_size
    batch_xs = X[start:end]
    batch_ys = Y[start:end]
    return batch_xs, batch_ys 

def vectorized_result(j):
    e = np.zeros((j.shape[0],10))
    for i in range(j.shape[0]):
        e[i,j[i]] = 1.0
    return e

def reporter(y_true, y_pred):
    print(classification_report(y_true, y_pred))

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):

    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(im,fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes) 
    plt.yticks(tick_marks, classes)
    print('Confusion matrix')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        cm[i,j]=round(cm[i,j],3)
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True fault type')
    plt.xlabel('CNN prediction fault type')

def cm_plot(y_, y, name, inds):
    class_names=np.array(['0','1','2','3','4','5','6','7','8','9'])
    cnf_matrix = confusion_matrix(y_, y)
    np.set_printoptions(precision=2)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion Matrix')
    cm_name = name + str(inds + 1) + ' of 10 folds'
    plt.title(cm_name)
    plt.savefig(cm_name)


X_train, y_train = load_mnist('MNIST_data', kind='train')

kf = KFold(n_splits = 10)
train_ind = []
test_ind = []
for train,test in kf.split(X_train):
    train_ind.append(train)
    test_ind.append(test)

X_train = np.array(X_train/255).astype(np.float32)
y_train = np.array(y_train)
y_vect = vectorized_result(y_train)

sess = tf.InteractiveSession()
                        
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu,
                    weights_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                    biases_initializer = tf.constant_initializer(0.1)):

    net = slim.conv2d(x_image, 32, [3, 3], scope = 'conv1', trainable = True)
    net = slim.max_pool2d(net, [2, 2], 2, scope = 'pool1', padding='SAME')
    net = slim.conv2d(net, 64, [3, 3], scope = 'conv2', trainable = True)
    net = slim.max_pool2d(net, [2, 2], 2, scope = 'pool2', padding='SAME')
    net_flat = slim.flatten(net, scope = 'flatten')
    fc1 = slim.fully_connected(net_flat, 1024, scope = 'fc1')
    keep_prob = tf.placeholder(tf.float32)
    fc1 = slim.dropout(fc1, keep_prob, is_training = True, scope = 'dropout1')
    fc2 = slim.fully_connected(fc1, 10, scope = 'fc2')


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=y_)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

pre_labels = tf.argmax(fc2,1)
correct_prediction = tf.equal(tf.argmax(fc2,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
total_acc = []
for j in range(10):
    X = X_train[train_ind[j]]
    Y_vect = y_vect[train_ind[j]]
    Tx = X_train[test_ind[j]]
    Ty = y_train[test_ind[j]]
    Ty_vect = y_vect[test_ind[j]]
    
    tf.global_variables_initializer().run()
    acc = []
    start = time.clock()
    for i in range(1000):
        batch_xs, batch_ys = generatebatch(X, Y_vect, 50, i)
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
##            acc.insert(0, train_accuracy)
##            if i > 1000:
##                if np.average(acc[0:10]) > 0.83 and acc[0] >= 0.85:
##                    break
    end = time.clock()
    print('training time:',str(end-start))
    labelss = pre_labels.eval(feed_dict= {x: Tx, y_: Ty_vect, keep_prob: 1.0})
    reporter(Ty, labelss)
    cm_plot(Ty, labelss,'confusion matrix ', j)
    now_accuracy = accuracy.eval(feed_dict={x: Tx, y_: Ty_vect, keep_prob: 1.0})
    total_acc.append(now_accuracy)
    print("test accuracy :",now_accuracy)

total_accuracy = np.average(total_acc)
print('total accuracy :',total_accuracy)




