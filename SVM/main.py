from sklearn import svm
import numpy as np
import tensorflow as tf
import time

if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data',one_hot=False)
    n = mnist.train.num_examples ##训练集数量
    m = mnist.test.num_examples ##测试集数量
    
    ##Load data
    train_data = mnist.train.images
    train_label = mnist.train.labels
    test_data = mnist.test.images
    test_label = mnist.test.labels
    ##

    start = time.process_time()

    predictor = svm.LinearSVC(C=2.0,tol=1e-3)
    predictor.fit(train_data[:n],train_label[:n])
    result = predictor.predict(test_data[:m])
    acc = np.sum(np.equal(result,test_label[:m]))/m
    print('accuracy = ',acc)
    print('Time used:',(time.process_time()-start))
    
