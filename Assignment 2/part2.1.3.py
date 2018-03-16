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

def CalculateWoptimal():
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')

    X = tf.placeholder(tf.float32, [None, 785], name='input_x')    
    W_optimal = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(X,X,transpose_a=True)),X,transpose_b=True),y_target)

    y_predicted = tf.matmul(X,W_optimal)
   
    
    return X,y_target,W_optimal,y_predicted

def calculateMSE():
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')
    X = tf.placeholder(tf.float32, [None, 785], name='input_x') 
    W = tf.placeholder(tf.float32,[785,1],name='optimal_w')
    y_predicted = tf.matmul(X,W)
    return W,X,y_target,y_predicted

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
weightdecay = 0.0
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


    epochs=(len(tempresult))
    result.append(tempresult)
     #calcuate the accuracy of valid data and test data for each epoch
    train_accuracy_list = []
    train_err, train_result = sess.run([cross_entropy_loss,y_predicted],feed_dict={x: trainData, y_target: trainTarget,weight_decay:weightdecay})
    train_result = sess.run(tf.sigmoid(train_result))
    for i in range(np.shape(train_result)[0]):
        train_accuracy_list.append(trainTarget[i] == (train_result[i]>0.5))
    logistic_train_accuracy = train_accuracy_list.count(True) / len (train_result) *100
    print("logistic_train_accuracy "+str(logistic_train_accuracy))

    valid_accuracy_list=[]
    valid_err, valid_result = sess.run([cross_entropy_loss,y_predicted],feed_dict={x: validData, y_target: validTarget,weight_decay:weightdecay})
    valid_result = sess.run(tf.sigmoid(valid_result))
    for i in range(np.shape(valid_result)[0]):
        valid_accuracy_list.append(validTarget[i] == (valid_result[i]>0.5))

    logistic_valid_accuracy = valid_accuracy_list.count(True) / len (valid_result) *100
    print("logistic_valid_accuracy "+str(logistic_valid_accuracy))


    test_accuracy_list = []
    test_err, test_result = sess.run([cross_entropy_loss,y_predicted],feed_dict={x: testData, y_target: testTarget,weight_decay:weightdecay})
    test_result = sess.run(tf.sigmoid(test_result))
    for i in range(np.shape(test_result)[0]):
        test_accuracy_list.append(testTarget[i] == (test_result[i]>0.5))
    logistic_test_accuracy = test_accuracy_list.count(True) / len (test_result) *100
    print("logistic_test_accuracy "+str(logistic_test_accuracy))



x_data,y_data,wnormal,y_norm_trainResult=CalculateWoptimal()
N=len(trainData)
trainData = np.reshape(trainData, [N, 28*28])
bias=tf.ones(shape=[N,1])
trainData=sess.run(tf.concat([trainData,bias],axis=1))
wNormalCal,y_norm_train=sess.run([wnormal,y_norm_trainResult],feed_dict={x_data:trainData, y_data: trainTarget})

ynorm_train_acc = []
for i in range(np.shape(y_norm_train)[0]):
    ynorm_train_acc.append(trainTarget[i] == (y_norm_train[i]>0.5))
Y_norm_train_accuracy = ynorm_train_acc.count(True) / len (ynorm_train_acc) *100
print("Y_norm_train_accuracy "+str(Y_norm_train_accuracy))

woptimal,x_data,y_data,y_norm_test_reuslt=calculateMSE()
N=len(testData)
testData = np.reshape(testData, [N, 28*28])
bias=tf.ones(shape=[N,1])
testData=sess.run(tf.concat([testData,bias],axis=1))
y_norm_test=sess.run([y_norm_test_reuslt],feed_dict={x_data:testData, y_data: testTarget,woptimal:wNormalCal})

y_norm_test=np.reshape(y_norm_test,[145,1])
ynorm_Test_acc = []
for i in range(np.shape(y_norm_test)[0]):
    ynorm_Test_acc.append(testTarget[i] == (y_norm_test[i]>0.5))
Y_norm_Test_accuracy = ynorm_Test_acc.count(True) / len (ynorm_Test_acc) *100
print("Y_norm_Test_accuracy "+str(Y_norm_Test_accuracy))

woptimal,x_data,y_data,y_norm_valid_reuslt=calculateMSE()
N=len(validData)
validData = np.reshape(validData, [N, 28*28])
bias=tf.ones(shape=[N,1])
validData=sess.run(tf.concat([validData,bias],axis=1))
y_norm_valid=sess.run([y_norm_valid_reuslt],feed_dict={x_data:validData, y_data: validTarget,woptimal:wNormalCal})
ynorm_Valid_acc = []

y_norm_valid=np.reshape(y_norm_valid,[100,1])

for i in range(np.shape(y_norm_valid)[0]):
    ynorm_Valid_acc.append(validTarget[i] == (y_norm_valid[i]>0.5))
Y_norm_Valid_accuracy = ynorm_Valid_acc.count(True) / len (ynorm_Valid_acc) *100
print("Y_norm_Valid_accuracy "+str(Y_norm_Valid_accuracy))



x = np.arange(epochs)
y = result[0]

 
# plot the cross entropy loss
#line_valid_error = plt.plot(x, valid_error_result,color='r', label="Valid Data Set Loss")
##line_test_accuracy = plt.plot(x, test_error_result,color='b', label="Test Data Set Loss")   

