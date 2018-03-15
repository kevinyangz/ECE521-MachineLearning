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
        crossEntropyLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target,logits=y_predicted))+ tf.divide(weight_decay,2)*tf.squeeze(tf.matmul(W,W,transpose_a=True))
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
        train = optimizer.minimize(loss=crossEntropyLoss)
        return W, b, X, y_target, y_predicted,learn_rate,weight_decay,crossEntropyLoss, train


trainData,trainTarget,validData,validTarget,testData,testTarget=Load_data()
w,b,x,y_target,y_predicted,learn_rate,weight_decay,cross_entropy_loss,train=build_graph()

#Reshape the data to N x 784 format
N=len(trainData)
trainData = np.reshape(trainData, [N, 28*28])
validData = np.reshape(validData, [len(validData), 28*28])
testData = np.reshape(testData, [len(testData), 28*28])

#print(trainTarget)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
learn= [0.005]
weightdecay = 0.01
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

         #calcuate the accuracy of valid data and test data for each epoch
        valid_accuracy_list = []
        valid_err, valid_result = sess.run([cross_entropy_loss,y_predicted],feed_dict={x: validData, y_target: validTarget,weight_decay:weightdecay})
        for i in range(np.shape(valid_result)[0]):
            valid_accuracy_list.append(validTarget[i] == (valid_result[i]>0.5))
        valid_accuracy = valid_accuracy_list.count(True) / len (valid_result) *100
        valid_accuracy_result.append(valid_accuracy)
        valid_error_result.append(valid_err)

        test_accuracy_list = []
        test_err, test_result = sess.run([cross_entropy_loss,y_predicted],feed_dict={x: testData, y_target: testTarget,weight_decay:weightdecay})
        for i in range(np.shape(test_result)[0]):
            test_accuracy_list.append(testTarget[i] == (test_result[i]>0.5))
        test_accuracy = test_accuracy_list.count(True) / len (test_result) *100
        test_accuracy_result.append(test_accuracy)
        test_error_result.append(test_err)

    epochs=(len(tempresult))
    result.append(tempresult)
    sess.run(init)
x = np.arange(epochs)
y = result[0]

line_valid_accuracy = plt.plot(x, valid_accuracy_result,color='r', label="Valid Data Set Accuracy")
line_test_accuracy = plt.plot(x, test_accuracy_result,color='g', label="Test Data Set Accuracy")

line_valid_error = plt.plot(x, valid_error_result,color='purple', label="Valid Data Set Loss")
line_test_accuracy = plt.plot(x, test_error_result,color='b', label="Test Data Set Loss")   
plt.legend(loc='best', shadow=True, fontsize='small')

    
plt.ylabel('Cross Entropy Loss/ Trainging Accuracy')
plt.xlabel('Number of Epochs')
plt.show()
