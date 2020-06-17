import numpy as np
import operator

m = 100 #测试数量，因为算法效率原因，不能测试所有的测试集

#训练集，训练标签，测试集，k值
def work(train_data,train_label,test_data,k):
    n = train_data.shape[0] #训练集的大小
    ret = np.empty([m],dtype=type(train_label[0])) #返回数组，返回一个数组，表示每个测试图片的标签
    for w in range(m):
        ##对测试集每个图片，让他和训练集计算距离矩阵
        item = test_data[w]
        rep_mat = np.tile(item,(n,1))
        rep_mat -= train_data
        rep_mat **= 2
        dis = rep_mat.sum(axis=1)
        dis **= 0.5 
        ##排序并找出k近邻
        index = dis.argsort()
        class_cnt = {}
        for i in range(k):
            label = train_label[index[i]]
            class_cnt[label] = class_cnt.get(label,0)+1
        class_count_list = sorted(class_cnt.items(), key=operator.itemgetter(1), reverse=True)
        ##得到第w个测试集的标签
        ret[w] = class_count_list[0][0]
    return ret

#单独测试knn算法
if __name__ == '__main__':
    train_data=np.array([[2.2,1.4],\
        [2.4,2.3],\
        [1.1,3.4],\
        [8.3,7.3],\
        [9.2,8.3],\
        [10.2,11.1],\
        [11.2,9.3],\
        [222,222]])
    train_label = ['A','A','A','B','B','B','B','C']
    test_data = np.array([[4.6,3.4],[10.3,10.2],[223,233]])
    print('ans_matrix = \n',work(train_data,train_label,test_data,3))




    