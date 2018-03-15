import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from random import *
import time

def Load_data():
    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
        #trainData is a tuple of size 3500
        return trainData,trainTarget,validData,validTarget,testData,testTarget


def CalculateWoptimal():
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')

    X = tf.placeholder(tf.float32, [None, 785], name='input_x')    
    W_optimal = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(X,X,transpose_a=True)),X,transpose_b=True),y_target)

    y_predicted = tf.matmul(X,W_optimal)
    meanSquaredError = tf.divide(tf.reduce_mean(tf.reduce_sum(tf.square(y_predicted - y_target), 
                                                        reduction_indices=1, 
                                                        name='squared_error'), 
                                          name='mean_squared_error'),2)
    
    return X,y_target,W_optimal,meanSquaredError
def calculateMSE():
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')
    X = tf.placeholder(tf.float32, [None, 785], name='input_x') 
    W = tf.placeholder(tf.float32,[785,1],name='optimal_w')
    y_predicted = tf.matmul(X,W)
    meanSquaredError = tf.divide(tf.reduce_mean(tf.reduce_sum(tf.square(y_predicted - y_target), 
                                                        reduction_indices=1, 
                                                        name='squared_error'), 
                                          name='mean_squared_error'),2)
    
    #accuracy_result=tf.sigmoid(y_predicted)
    
    return W,X,y_target,meanSquaredError,y_predicted

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

trainData,trainTarget,validData,validTarget,testData,testTarget=Load_data()
x_data,y_data,wnormal,mse=CalculateWoptimal()
#Reshape the data to N x 784 format
print("Normal Equation Part")
start_time = int(round(time.time() * 1000))

N=len(trainData)
trainData = np.reshape(trainData, [N, 28*28])
bias=tf.ones(shape=[N,1])
trainData=sess.run(tf.concat([trainData,bias],axis=1))
Train_meanSquaredError,wNormalCal=sess.run([mse,wnormal],feed_dict={x_data:trainData, y_data: trainTarget})
#print("TrainData MSE:"+str(Train_meanSquaredError))


woptimal,x_data,y_data,mse,accuracy=calculateMSE()
N=len(testData)
testData = np.reshape(testData, [N, 28*28])
bias=tf.ones(shape=[N,1])
testData=sess.run(tf.concat([testData,bias],axis=1))
test_meanSquaredError,accuracy_result=sess.run([mse,accuracy],feed_dict={x_data:testData, y_data: testTarget,woptimal:wNormalCal})
end_time = int(round(time.time() * 1000))
run_time=end_time-start_time
print("runing time for NormalEquation is %sms"%(run_time))
accuracy_list = []

for i in range(np.shape(accuracy_result)[0]):
    accuracy_list.append(testTarget[i]==(accuracy_result[i]>0.5))
accuracy = accuracy_list.count(True) / len(accuracy_result) * 100
print("Accuracy for the validation set is %s%s,MeanSquarederror is %s"%(accuracy,'%',test_meanSquaredError))





