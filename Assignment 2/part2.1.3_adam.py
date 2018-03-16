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
        y_predicted = tf.matmul(X,W) + b

        #crossEntropyLoss = tf.reduce_sum(-y_target*log(y_predicted)-(1-y_target)*log(1-y_predicted), 1)  + tf.divide(weight_decay,2)*tf.squeeze(tf.matmul(W,W,transpose_a=True))
        crossEntropyLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target,logits=y_predicted))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
        train = optimizer.minimize(loss=crossEntropyLoss)
        return W, b, X, y_target, y_predicted,learn_rate,weight_decay,crossEntropyLoss, train


def build_graph_adam():
    W = tf.Variable(tf.truncated_normal(shape=[784,1], stddev=0.5), name='weights_adam') #not too sure about stddev
    b = tf.Variable(0.0, name='biases_adam')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x_adam')
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y_adam')
    learn_rate=tf.placeholder(tf.float32,shape=[],name='learn_rate_adam')
    weight_decay = tf.placeholder(tf.float32,shape=[],name='weight_decay_adam')

    # Graph definition
    y_predicted = tf.matmul(X,W) + b

    MSE = tf.divide(tf.reduce_mean(tf.reduce_sum(tf.square(y_predicted - y_target),\
                                                              reduction_indices=1,\
                                                              name='squared_error'),\
                                                name='mean_squared_error'),2)
    #crossEntropyLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target,logits=y_predicted))+ tf.divide(weight_decay,2)*tf.squeeze(tf.matmul(W,W,transpose_a=True))

    optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)
    train = optimizer.minimize(loss=MSE)
    return W, b, X, y_target, y_predicted,learn_rate,weight_decay,MSE, train


trainData,trainTarget,validData,validTarget,testData,testTarget=Load_data()
w,b,x,y_target,y_predicted,learn_rate,weight_decay,cross_entropy_loss,train=build_graph()
w_adam,b_adam,x_adam,y_target_adam,y_predicted_adam,learn_rate_adam,weight_decay_adam,mse_loss_adam,train_adam=build_graph_adam()

#Reshape the data to N x 784 format
N=len(trainData)
trainData = np.reshape(trainData, [N, 28*28])
validData = np.reshape(validData, [len(validData), 28*28])
testData = np.reshape(testData, [len(testData), 28*28])

#print(trainTarget)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
weightdecay = 0
result=[]
valid_accuracy_result = []
valid_error_result = []
test_accuracy_result = []
test_error_result = []
weight_list = []
bias_list = []
#valid_mse_list = []
epochs=0
iterations = 5000
num_epochs = int(iterations/7)
logistic_accuracy_list = []
linear_accuracy_list = []
logistic_error_list = []
linear_error_list =[]




for optimizer in range(0,2):
    learnrate=0.001
    for step in range(0,num_epochs):
        for i in range(0,7):
            start_index = i* 500
            minix=trainData[start_index:start_index+500]
            miniy=trainTarget[start_index:start_index+500]
            #logistic regression
            if(optimizer == 0):
                _, err, currentW, currentb, yhat = sess.run([train, cross_entropy_loss, w, b, y_predicted], feed_dict={x: minix, y_target: miniy,learn_rate:learnrate, weight_decay:weightdecay})
            #linear regression
            else:
                _, err, currentW, currentb, yhat = sess.run([train_adam, mse_loss_adam, w_adam, b_adam, y_predicted_adam], feed_dict={x_adam: minix, y_target_adam: miniy,learn_rate_adam:learnrate, weight_decay_adam:weightdecay})

        if (optimizer == 0):
            #add error and the into a list  here
            train_accuracy_list = []
            train_err, train_result = sess.run([cross_entropy_loss,y_predicted],feed_dict={x: trainData, y_target: trainTarget,weight_decay:weightdecay})
            logistic_error_list.append(train_err)
            train_result = sess.run(tf.sigmoid(train_result))
            for i in range(np.shape(train_result)[0]):
                train_accuracy_list.append(trainTarget[i] == (train_result[i]>0.5))
            logistic_train_accuracy = train_accuracy_list.count(True) / len (train_result) *100
            logistic_accuracy_list.append(logistic_train_accuracy)

        else:
            #add error and the into a list  here
            train_accuracy_list = []
            train_err, train_result = sess.run([mse_loss_adam,y_predicted_adam],feed_dict={x_adam: trainData, y_target_adam: trainTarget,weight_decay_adam:weightdecay})
            linear_error_list.append(train_err)
            for i in range(np.shape(train_result)[0]):
                train_accuracy_list.append(trainTarget[i] == (train_result[i]>0.5))
            linear_train_accuracy = train_accuracy_list.count(True) / len (train_result) *100
            linear_accuracy_list.append(linear_train_accuracy)
    sess.run(init)

epochs=(len(logistic_error_list))
#plotting

x = np.arange(epochs)

# plot the cross entropy loss
line_valid_error = plt.plot(x, logistic_error_list,color='r', label="Logistic Regression Train Data Cross Entropy Error")
line_test_accuracy = plt.plot(x, linear_error_list,color='b', label="Linear Regression Train Data Cross Entropy Error")
plt.legend(loc='best', shadow=True, fontsize='small')
plt.ylabel('Loss(Cross Entropy for Logistic, MSE for Linear)')
plt.xlabel('Number of Epochs')
plt.show()

# plot the validation accuracy
line_valid_accuracy = plt.plot(x, logistic_accuracy_list,color='r', label="Logistic Regression Train Data Set Accuracy")
line_test_accuracy = plt.plot(x, linear_accuracy_list,color='g', label="Linear Regression Train Data Set Accuracy")
plt.legend(loc='best', shadow=True, fontsize='small')
plt.ylabel('Classification Accuracy')
plt.xlabel('Number of Epochs')
plt.show()


