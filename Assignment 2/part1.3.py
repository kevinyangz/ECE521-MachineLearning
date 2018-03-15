import numpy as np
import tensorflow as tf
from random import *

#Sample code for loading data

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

def build_graph():
        W = tf.Variable(tf.truncated_normal(shape=[784,1], stddev=0.5), name='weights') #not too sure about stddev
        b = tf.Variable(0.0, name='biases')
        X = tf.placeholder(tf.float32, [None, 784], name='input_x')
        y_target = tf.placeholder(tf.float32, [None,1], name='target_y')
        learn_rate=tf.placeholder(tf.float32,shape=[],name='learn_rate')
        weight_decay = tf.placeholder(tf.float32,shape=[],name='weight_decay')
        # Graph definition
        y_predicted = tf.matmul(X,W) + b
     
        #meanSquaredError = tf.reduce_mean(tf.reduce_sum((y_predicted - y_target)**2, 1))/2
        weightDecayLoss =  tf.reduce_mean(tf.reduce_sum((y_predicted - y_target)**2, 1))/2 + tf.divide(weight_decay,2)*tf.squeeze(tf.matmul(W,W,transpose_a=True))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
        train = optimizer.minimize(loss=weightDecayLoss)
        return W, b, X, y_target, y_predicted,learn_rate,weight_decay,weightDecayLoss,train


        
        
        
trainData,trainTarget,validData,validTarget,testData,testTarget=Load_data()
w,b,x,y_target,y_predicted,learn_rate,weight_decay,weight_decay_loss, train=build_graph()

#Reshape the data to N x 784 format
N=len(trainData)
trainData = np.reshape(trainData, [N, 28*28])
validData = np.reshape(validData, [len(validData), 28*28])
#print(trainTarget)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
learnRate= 0.005
weight_decay_list = [0,0.001,0.1,1]

valid_weight_decay_list = []
epochs=0
iterations = 20000
num_epochs = int(iterations/7)
for weightDecay in weight_decay_list:
    tempresult=[]
    for step in range(0,num_epochs):
        for i in range(0,7):
            start_index = i* 500
            minix=trainData[start_index:start_index+500]
            miniy=trainTarget[start_index:start_index+500]
            _, err, currentW, currentb, yhat = sess.run([train, weight_decay_loss, w, b, y_predicted], feed_dict={x: minix, y_target: miniy,learn_rate:learnRate,weight_decay:weightDecay})
        tempresult.append(err)
    epochs=(len(tempresult))
    valid_err = sess.run(weight_decay_loss,feed_dict={x: validData, y_target: validTarget,weight_decay:weightDecay})
    print("Weight Decay Coefficient " + str(weightDecay) + ": " + "valid data error: " + str(valid_err) + " test data error: " + str(err))
    valid_weight_decay_list.append(valid_err)
    sess.run(init)





