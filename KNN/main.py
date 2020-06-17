import knn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    n = mnist.train.num_examples
    m = knn.m 

    ###从tensorflow中得到mnist数据集，并把向量标签转化为数字标签，便于map操作
    train_data = mnist.train.images
    train_label = mnist.train.labels
    train_labels = np.empty([n],dtype=np.int32)
    for i in range(n):
        for k in range(10):
            if train_label[i][k] == 1:
                train_labels[i] = k
    
    test_data = mnist.test.images
    test_label = mnist.test.labels
    test_labels = np.empty([m],dtype=np.int32)
    for i in range(m):
        for k in range(10):
            if test_label[i][k] == 1:
                test_labels[i] = k
    
    ###测试1到7的K值，并计算准确率
    for K in range(1,8):
        knn_labels = knn.work(train_data,train_labels,test_data,K)
        
        cnt = 0
        for i in range(m):
            if knn_labels[i] == test_labels[i]:
                cnt = cnt+1
        print('KNN Algorithm:')
        print('When K=',K,'  ','Accuracy=',cnt/m)



