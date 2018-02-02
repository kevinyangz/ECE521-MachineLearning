import os
import matplotlib
matplotlib.use("Agg")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from part2 import findResponsibility


#part3 Load Data
def data_segmentation(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255
    data = np.reshape(data, [-1, 32*32])
    target = np.load(target_path)
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
    data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
    data[rnd_idx[trBatch + validBatch+1:-1],:]
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
    target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
    target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def findPredictClassLableAndAccuracy(trainMatrix,trainTarget,testMatrix,testTarget,k):
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    responsibilityMatrix=findResponsibility(trainMatrix,testMatrix,k)
    result=(tf.transpose(sess.run(responsibilityMatrix)))
    
    shape=tf.shape(result)
    row=shape[0]
    col=shape[1]
    #print (responsibilityMatrix)
    distance_bool = (result > 0)
    trainTarget=tf.reshape(trainTarget,[1,-1])
    
    tileresult=tf.reshape(tf.tile(trainTarget,[1,row]),[row,col])
    maskresult=tf.boolean_mask(tileresult,distance_bool)
    result=sess.run(tf.reshape((maskresult),[row,k]))
    
    prediction_result = []
    accuracy_result = []
    
    for i in range(np.shape(result)[0]):
        #print("-------------------")
        result_i,indx_i,count_i = tf.unique_with_counts(result[i])
        top_val,top_index = tf.nn.top_k(result_i, 1)
        top_result_i = tf.gather(result_i,top_index)
        accuracy_result.append(testTarget[i] == sess.run(top_result_i)[0])
        prediction_result.append(sess.run(top_result_i)[0])
    
    #print(accuracy_result)
    #print(prediction_result)
    #print(testTarget)
    return prediction_result, accuracy_result

def main():
        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)

        k_list = [1, 5, 10, 25, 50, 100, 200]
        accuracy_list = []
        trainData, validData, testData, trainTarget, validTarget, testTarget=data_segmentation("data.npy","target.npy",0)

        for k in k_list:
            prediction_result, accuracy_result = findPredictClassLableAndAccuracy(trainData,trainTarget,validData,validTarget, k)
            accuracy = accuracy_result.count(True) / len(accuracy_result) * 100
            accuracy_list.append(accuracy)
            #print(testTarget)
            print("Accuracy for the validation set is %s %s,for k = %s"%(accuracy,'%',k))

        k_best = k_list[np.argmax(accuracy_list)]
        print("The best k for the validation set is: %s"%k_best)

        prediction_result, accuracy_result = findPredictClassLableAndAccuracy(trainData,trainTarget,testData,testTarget,k_best)
        accuracy = accuracy_result.count(True) / len(accuracy_result) * 100
        print("Accuracy for the test set is: %s %s, for k = %s"%(accuracy,'%',k_best))

        print("For k=10 display one failure case: test image and the 10 nearest images")
        prediction_result, accuracy_result = findPredictClassLableAndAccuracy(trainData,trainTarget,testData,testTarget,10)
        first_fail_index = accuracy_result.index(False)
        #first_fail_index = 6

        shape=tf.shape(testData)
        col=shape[1]
        first_fail = tf.slice(testData, [first_fail_index, 0], [1,col])
        #first_fail = tf.reshape(testData[first_fail_index], [1, -1])
        #print(sess.run(first_fail))
        #print(first_fail_index)

        responsibilityMatrix = findResponsibility(trainData,first_fail,10)
        responsibilityMatrix = tf.reshape(responsibilityMatrix,[-1])
        #print(sess.run(tf.size(trainData)))
        #print(sess.run(tf.size(first_fail)))
        #print(sess.run(tf.size(responsibilityMatrix)))
        #print(sess.run(responsibilityMatrix))

        closest_10_index = tf.where((responsibilityMatrix>0))
        closest_10_index = sess.run(closest_10_index)
        #print(closest_10_index)

        tmp = testData[first_fail_index]
        tmp = tf.cast(tmp, tf.float32)
        tmp = tf.reshape(tmp,[32,32])

        plt.figure()
        plt.subplot(3,5,1)
        plt.imshow(sess.run(tmp),cmap=plt.gray())

        j = 5
        for i in closest_10_index:
            j = j + 1
            tmp = trainData[i]
            tmp = tf.cast(tmp, tf.float32)
            tmp = tf.reshape(tmp,[32,32])
            plt.subplot(3,5,j)
            plt.imshow(sess.run(tmp),cmap=plt.gray())

        plt.show()



if __name__ == "__main__":
    main()



