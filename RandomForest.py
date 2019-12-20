import numpy as np
import pandas as pd
import struct
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection

# Written by Hastings (Shihao Li)

# Load dataset
def loadImageSet(filename):
 
    binfile = open(filename, 'rb') 
    buffers = binfile.read()
 
    head = struct.unpack_from('>IIII', buffers, 0) 
 
    offset = struct.calcsize('>IIII')  
    num = head[1]
    width = head[2]
    height = head[3]
 
    bits = num * width * height  
    bitsString = '>' + str(bits) + 'B'  # fmt：'>47040000B'
 
    imgs = struct.unpack_from(bitsString, buffers, offset) 
 
    binfile.close()
    imgs = np.reshape(imgs, [num, width * height]) # reshape to [60000,784]
 
    return imgs

def loadLabelSet(filename):
 
    binfile = open(filename, 'rb') 
    buffers = binfile.read()
 
    head = struct.unpack_from('>II', buffers, 0) 
 
    num = head[1]
    offset = struct.calcsize('>II')  
 
    numString = '>' + str(num) + "B" 
    labels = struct.unpack_from(numString, buffers, offset) 
 
    binfile.close()
    labels = np.reshape(labels, [num]) 
 
    return labels

data_test = loadImageSet('t10k-images-idx3-ubyte')
label_test= loadLabelSet('t10k-labels-idx1-ubyte')

data_train = loadImageSet('train-images-idx3-ubyte')
label_train = loadLabelSet('train-labels-idx1-ubyte')

# Precessing
# transform type
data_test.astype(np.uint8)
data_train.astype(np.uint8)
# compress color level 256->16
data_train = data_train // 16
data_test = data_test // 16

# rf = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=110,
                                #   min_samples_leaf=20, max_features='sqrt' ,oob_score=True, random_state=10)

def XFordCrossValidation(n_Ford):
    data_all = np.concatenate([data_train, data_test])
    label_all = np.concatenate([label_train, label_test])

    rf = RandomForestClassifier(n_estimators= 10, n_jobs= 5)    # Random Forest
    sss = model_selection.KFold(n_splits=n_Ford,shuffle=True)   # KFold

    time_total = 0
    ave_accuracy = 0

    for train_index, test_index in sss.split(data_all, label_all):
        X_train, X_test = data_all[train_index], data_all[test_index]
        y_train, y_test = label_all[train_index], label_all[test_index]

        time_start = time.time()
        rf.fit(X_train,y_train)
        y_pred = rf.predict(X_test)
        time_end = time.time()
        time_total += time_end - time_start

        accuracy = metrics.balanced_accuracy_score(y_test, y_pred)
        ave_accuracy += accuracy
        result_metrix = metrics.classification_report(y_test, y_pred)
        confusion_metrix = pd.crosstab(y_pred, y_test, rownames=['True'], colnames=['Predicted'], margins=True)
        metrics.roc_curve
        print('New Fold:')
        print('Accurancy = ', accuracy)
        print(confusion_metrix)
        print(result_metrix)

    print('Training Time: %.3f' %(time_total/10))
    ave_accuracy = ave_accuracy/n_Ford
    print('Training Accuracy %.3f' %ave_accuracy)
    
XFordCrossValidation(10)


# def normaltest():
#     time_start = time.time()
#     rf = RandomForestClassifier(n_estimators= 10, n_jobs= 3)
#     rf.fit(data_train, label_train)
#     label_predict = rf.predict(data_test)
#     time_end = time.time()

#     print('Training Time: %.3f' %(time_end-time_start))
#     print('Accuracy: %.2f%%' %(100*metrics.accuracy_score(label_test, label_predict)))

#     result_metrix = metrics.classification_report(label_test, label_predict)
#     confusion_metrix = pd.crosstab(label_predict, label_test, rownames=['True'], colnames=['Predicted'], margins=True)
#     print(confusion_metrix)
#     print(result_metrix)

# normaltest()