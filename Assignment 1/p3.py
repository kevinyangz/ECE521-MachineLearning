import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter

def euclideanDistance(x, z):
    y = x-z
    distance = tf.transpose(y)*(y)
    return distance

def euclideanDistanceMultipleDimension(A,B):
    r = tf.reduce_sum(A*A, 1)
    d = tf.reduce_sum(B*B,1)
    d = tf.reshape(d, [-1, 1])
    #print (d.shape)
    r = tf.reshape(r, [-1, 1])
    
    #print (sess.run(r))
    
    #print (sess.run(d))
    
    #print (sess.run(2*tf.matmul(A, tf.transpose(B))))
    D = r - 2*tf.matmul(A, tf.transpose(B)) + tf.transpose(d)
    return D

def euclideanV3(x,z):
    #get the dimension of z,x
    z_dim = tf.shape(z) #z_dim[0] = num of rows, z_dim[1] = num of columns
    z_row = z_dim[0]
    z_col = z_dim[1]
    x_dim = tf.shape(x)
    x_row = x_dim[0]
    x_col = x_dim[1]
    
    #duplicate each row in x by z_row times
    x_expanded = tf.tile(x,[1,z_row],name='x_expanded') #note this is a flattend array
    #print (sess.run(x_expanded))
    # get the desired shape
    x_reshaped = tf.reshape(x_expanded,[x_row,z_row,z_col])
    #print (sess.run(x_reshaped))
    
    #get the square of the difference, tensorflow broadcasts z to match the size of s
    square_difference = tf.square(x_reshaped-z)
    #reduce sum each row
    #print (sess.run(square_difference))
    d = tf.reduce_sum(square_difference,2)
    return d


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

def findResponsibility(trainMatrix,testMatrix,k):
    distance = euclideanDistanceMultipleDimension(trainMatrix,testMatrix)
    distanceVector = tf.transpose(distance)
    distanceShape = tf.shape(distance)
    smallestIndices = tf.nn.top_k(-distanceVector, k).indices
    hotMatrix = tf.one_hot(smallestIndices, depth=distanceShape[0], on_value = 1/k)
    responsibilityMatrix = tf.transpose(tf.reduce_max(hotMatrix,1))
    responsibilityMatrix = tf.cast(responsibilityMatrix, tf.float64)
    return responsibilityMatrix

def findPredictClassLableAndAccuracy(trainMatrix,trainTarget,testMatrix,testTarget,k):
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
    
    print(accuracy_result)
    print(prediction_result)
    return prediction_result, accuracy_result


sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

k_list = [1, 5, 10, 25, 50, 100, 200]
accuracy_list = []
trainData, validData, testData, trainTarget, validTarget, testTarget=data_segmentation("data.npy","target.npy",0)

if False:
    for k in k_list:
        prediction_result, accuracy_result = findPredictClassLableAndAccuracy(trainData,trainTarget,validData,validTarget, k)
        accuracy = accuracy_result.count(True) / len(accuracy_result) * 100
        accuracy_list.append(accuracy)
        print(testTarget)
        print("Accuracy for the validation set is %s %s,for k = %s"%(accuracy,'%',k))

    k_best = k_list[np.argmax(accuracy_list)]
    print("The best k for the validation set is: %s"%k_best)

    prediction_result, accuracy_result = findPredictClassLableAndAccuracy(trainData,trainTarget,testData,testTarget,k_best)
    accuracy = accuracy_result.count(True) / len(accuracy_result) * 100
    print("Accuracy for the test set is: %s %s, for k = %s"%(accuracy,'%',k_best))

print("For k=10 display one failure case: test image and the 10 nearest images")
prediction_result, accuracy_result = findPredictClassLableAndAccuracy(trainData,trainTarget,testData,testTarget,10)
first_fail_index = accuracy_result.index(False)
print(testData[first_fail_index])
print(first_fail_index)

#not sure yet, trying to display image
tmp = testData[first_fail_index]
tmp = tf.cast(tmp, tf.float32)
tmp = tf.reshape(tmp,[32,32])

plt.imshow(sess.run(tmp),cmap=plt.gray())
plt.show()




