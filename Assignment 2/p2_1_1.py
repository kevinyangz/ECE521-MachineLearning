import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
        y_predicted = tf.sigmoid(tf.matmul(X,W) + b)
     
        #crossEntropyLoss = tf.reduce_sum(-y_target*log(y_predicted)-(1-y_target)*log(1-y_predicted), 1)  + tf.divide(weight_decay,2)*tf.squeeze(tf.matmul(W,W,transpose_a=True))
        crossEntropyLoss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target,logits=y_predicted)+ tf.divide(weight_decay,2)*tf.squeeze(tf.matmul(W,W,transpose_a=True))
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
        train = optimizer.minimize(loss=crossEntropyLoss)
        return W, b, X, y_target, y_predicted,learn_rate,weight_decay,crossEntropyLoss, train


trainData,trainTarget,validData,validTarget,testData,testTarget=Load_data()
w,b,x,y_target,y_predicted,learn_rate,weight_decay,cross_entropy_loss,train=build_graph()

#Reshape the data to N x 784 format
N=len(trainData)
trainData = np.reshape(trainData, [N, 28*28])
#validData = np.reshape(validData, [len(validData), 28*28])
#print(trainTarget)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
learn= [0.005,0.001,0.0001]
weightdecay = 0.01
result=[]
weight_list = []
bias_list = []
#valid_mse_list = []
epochs=0
iterations = 5000
num_epochs = int(iterations/7)
for rate in learn:
    #w,b,x,y_target,y_predicted,learn_rate,mse,train=build_graph()
    learnrate=rate
    tempresult=[]
    for step in range(0,num_epochs):
        for i in range(0,7):
            start_index = i* 500
            minix=trainData[start_index:start_index+500]
            miniy=trainTarget[start_index:start_index+500]
            _, err, currentW, currentb, yhat = sess.run([train, cross_entropy_loss, w, b, y_predicted], feed_dict={x: minix, y_target: miniy,learn_rate:learnrate, weight_decay:weightdecay})
        tempresult.append(err)
    epochs=(len(tempresult))
    result.append(tempresult)
    weight_list.append(currentW)
    bias_list.append(currentb)
    #valid_err = sess.run(mse,feed_dict={x: validData, y_target: validTarget})
    #print("Learning Rate " + str(rate) + ": " + str(valid_err))
    #valid_mse_list.append(valid_err)
    sess.run(init)
x = np.arange(epochs)
y = result[0]
color=['r','g','b']
for idx, val in enumerate(learn):
    line, = plt.plot(x, result[idx],color=color[idx], label="learning rate: "+str(val))
    
plt.legend(loc='upper right', shadow=True, fontsize='x-large')

    
plt.ylabel('Training loss')
plt.xlabel('Number of Epochs')
plt.show()
