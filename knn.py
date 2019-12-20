import gzip
import os
from sklearn.model_selection import KFold
import numpy as np
import copy
import operator
import datetime

start = datetime.datetime.now()

def load_mnist(path, kind='train'):

    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        
    return images, labels

image, labels = load_mnist('MNIST_data', kind='train')
# 加载数据
image = np.array(image/16).astype(np.uint8)
labels = np.array(labels)
# 数据处理（图像像素除以16，这个原来就用的除以十六的方法，效果不错）
kf = KFold(n_splits = 10)
train_ind = []
test_ind = []
for train,test in kf.split(image):
    train_ind.append(train)
    test_ind.append(test)
# 生成交叉验证集

# 0， 256 二值法
# 去噪点

# def removeNoise(dataArr):
#     wide = dataArr.shape[0]

#     for i in range(wide):
#         for j in range(wide):



# for i in range(num_data):
#     for j in range(wide_data):
#         for k in range(wide_data):
#             if data[i][j][k] == 0:
#                 continue

#             noise = 0
#             if j == 0:
#                 noise += 1
#             else:
#                 if  data[i][j-1][k] == 0:
#                     noise += 1

#             if j == wide_data - 1:
#                 noise += 1
#             else:
#                 if  data[i][j+1][k] == 0:
#                     noise += 1

#             if k == 0:
#                 noise += 1
#             else:
#                 if  data[i][j][k-1] == 0:
#                     noise += 1

#             if k == wide_data - 1:
#                 noise += 1
#             else:
#                 if  data[i][j][k+1] == 0:
#                     noise += 1
#             if k == 4:
#                 data[i][j][k] == 0
#             if k > 4:
#                 print('error NOISE')

# for i in range(num_data_test):
#     for j in range(wide_data):
#         for k in range(wide_data):
#             if data_test[i][j][k] == 0:
#                 continue

#             noise = 0
#             if j == 0:
#                 noise += 1
#             else:
#                 if  data_test[i][j-1][k] == 0:
#                     noise += 1

#             if j == wide_data - 1:
#                 noise += 1
#             else:
#                 if  data_test[i][j+1][k] == 0:
#                     noise += 1

#             if k == 0:
#                 noise += 1
#             else:
#                 if  data_test[i][j][k-1] == 0:
#                     noise += 1

#             if k == wide_data - 1:
#                 noise += 1
#             else:
#                 if  data_test[i][j][k+1] == 0:
#                     noise += 1
#             if k == 4:
#                 data_test[i][j][k] == 0
#             if k > 4:
#                 print('error NOISE')

# def removeNoise(datalist, len):
#     for i in range(len):
#         for j in range(wide_data):
#             for k in range(wide_data):
#                 if datalist[i][j][k] != 0 and datalist[i][j][k] == 0 and datalist[i][j][k] == 0 and datalist[i][j][k] == 0 and datalist[i][j][k] == 0:
#                     datalist[i][j][k] = 0
#                     print('A NOISE')

# removeNoise(data,num_data)
# removeNoise(data_test, num_data_test)

# 图片变形


# 归 1

# for i in range(num_data):
#     for j in range(wide_data**2):
#             if data[i][j] > 0:
#                 data[i][j] = 1.

# for i in range(num_data_test):
#     for j in range(wide_data**2):
#             if data_test[i][j] > 0:
#                 data_test[i][j] = 1. 

# for i in range(num_data):
#     for j in range(wide_data**2):
#             if data[i][j] > 30:
#                 data[i][j] = np.uint8(1)
#             else:
#                 data[i][j] = np.uint8(0)

# for i in range(num_data_test):
#     for j in range(wide_data**2):
#             if data_test[i][j] > 30:
#                 data_test[i][j] = np.uint8(1) 
#             else:
#                 data_test[i][j] = np.uint8(0)




# for i in range(num_data):
#     average = np.mean(data[i])
#     for j in range(wide_data**2):

#             if data[i][j] > average:
#                 data[i][j] = 1.
#             else:
#                 data[i][j] = 0.

# for i in range(num_data_test):
#     average = np.mean(data_test[i])
#     for j in range(wide_data**2):
#             if data_test[i][j] > average:
#                 data_test[i][j] = 1. 
#             else:
#                 data_test[i][j] = 0.


# print(label.shape)

# temp = np.zeros((num_data * 2, wide_data * wide_data))

# for i in range(num_data):
#     for j in range(wide_data**2):
#             if data[i][j] > 30:
#                 temp[i][j] = 1.
#             else:
#                 temp[i][j] = 0
#                 temp[i+num_data][j] = 1.
# data = temp

# label = np.concatenate((label, label), axis = 0)


# for i in range(num_data_test):
#     for j in range(wide_data**2):
#             if data_test[i][j] > 30:
#                 data_test[i][j] = 1. 
#             else:
#                 data_test[i][j] = 0.

# 上面一大堆注释掉的方法好像是各种预处理，我都没用

# testnumber = data_test.shape[0]
# 获得测试集的个数
precision = np.zeros((10,10))
recall = np.zeros((10,10))
cf_matrix = np.zeros((10,10))
total_accuracy = []
# 初始化各参数
for m in range(10):
    data = image[train_ind[m]]
    label = labels[train_ind[m]]
    data_test = image[test_ind[m]]
    label_test = labels[test_ind[m]]
    testnumber = data_test.shape[0]
    # 读取交叉训练集和验证集
    
    kernal = 5
    # 临近值选取5（唯一可调参数）
    accurancy = 0
    acc = np.zeros((10,10))
    for i in range(testnumber):
        dis = np.sum((data - data_test[i,:])**2, axis=1)
        arg_descending = np.argsort(dis) #argsort(从小到大索引)

        nearest_labels = {}

        for k in range(kernal):
            label_of_k = label[arg_descending[k]]
            nearest_labels[label_of_k] = nearest_labels.get(label_of_k, 0) + 1
        sorted_nearest_labels = sorted(nearest_labels.items(), key=operator.itemgetter(1), reverse=True)
        predict_label = sorted_nearest_labels[0][0]
        acc[int(predict_label)][int(label_test[i])] += 1
        if predict_label == label_test[i]:
            accurancy = accurancy + 1
    
    cf_matrix += acc
    for j in range(10):
        precision[m][j] = acc[j][j]/acc[j].sum()
        recall[m][j] = acc[j][j]/acc[:,j].sum()
    
    print('Kernal =', kernal ,'accurancy ==',accurancy/testnumber * 100, '%')
    total_accuracy.append(accurancy/testnumber * 100)

cf_matrix = cf_matrix/10
print('precision of label 1:',precision[:,0].sum()/10,'     ','recall of label 1:',recall[:,0].sum()/10)
print('precision of label 2:',precision[:,1].sum()/10,'     ','recall of label 2:',recall[:,1].sum()/10)
print('precision of label 3:',precision[:,2].sum()/10,'     ','recall of label 3:',recall[:,2].sum()/10)
print('precision of label 4:',precision[:,3].sum()/10,'     ','recall of label 4:',recall[:,3].sum()/10)
print('precision of label 5:',precision[:,4].sum()/10,'     ','recall of label 5:',recall[:,4].sum()/10)
print('precision of label 6:',precision[:,5].sum()/10,'     ','recall of label 6:',recall[:,5].sum()/10)
print('precision of label 7:',precision[:,6].sum()/10,'     ','recall of label 7:',recall[:,6].sum()/10)
print('precision of label 8:',precision[:,7].sum()/10,'     ','recall of label 8:',recall[:,7].sum()/10)
print('precision of label 9:',precision[:,8].sum()/10,'     ','recall of label 9:',recall[:,8].sum()/10)
print('precision of label 10:',precision[:,9].sum()/10,'    ','recall of label 10:',recall[:,9].sum()/10)

print('confusion matrix:')
print(cf_matrix)
total_accuracy = np.array(total_accuracy)
print('average accuracy:', total_accuracy.sum()/10)

end = datetime.datetime.now()
print('total time:', (end-start).seconds)



# rank 1 accurancy ==  71.2 %
# rank 3 accurancy ==  71.45 %
# rank 5 accurancy ==  71.89 %
# rank 10 accurancy ==  72.6 %
# rank 11 accurancy ==  72.3 %
# rank 13 accurancy ==  71.95 %
# rank 15 accurancy ==  71.95 %
# rank 20 accurancy ==  71.25 %
# rank 21 accurancy ==  71.39 %

# 0->0, +->1
# Kernal = 1 accurancy == 82.2 %
# Kernal = 3 accurancy == 83.5 %
# Kernal = 5 accurancy == 83.1 %
# Kernal = 10 accurancy == 82.3 %
# Kernal = 11 accurancy == 82.2 %
# Kernal = 13 accurancy == 81.75 %
# Kernal = 15 accurancy == 81.55 %
# Kernal = 20 accurancy == 81.45 %
# Kernal = 21 accurancy == 80.3 %


#  -> / 30,

# Kernal = 1 accurancy == 83.55 %
# Kernal = 3 accurancy == 84.15 %
# Kernal = 5 accurancy == 84.65 %
# Kernal = 10 accurancy == 84.25 %


# //16
# Kernal = 1 accurancy == 82.9 %
# Kernal = 3 accurancy == 83.95 %
# Kernal = 5 accurancy == 85.53 %
# Kernal = 10 accurancy == 83.7 %
# Kernal = 11 accurancy == 83.4 %
# Kernal = 13 accurancy == 82.8 %
# Kernal = 15 accurancy == 83.1 %
# Kernal = 20 accurancy == 82.89 %
# Kernal = 21 accurancy == 82.75 %
