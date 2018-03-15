import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def Load_NotMinst():
    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
        return trainData,trainTarget,validData,validTarget,testData,testTarget
def Build_Graph():

    W = tf.Variable(tf.truncated_normal(shape=[784,10], stddev=0.5), name='weights') #not too sure about stddev
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,10], name='target_y')
    learn_rate=tf.placeholder(tf.float32,shape=[],name='learn_rate')
    # Graph definition
    y_predicted = tf.matmul(X,W) + b
    crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target,logits=y_predicted))+ tf.divide(0.01,2)*tf.nn.l2_loss(W)
    optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)
    train = optimizer.minimize(loss=crossEntropyLoss)
    return crossEntropyLoss,W,X,y_target,train,learn_rate

trainMinstData,trainMinstTarget,validMinstData,validMinstTarget,testMinstData,testMinstTarget= Load_NotMinst()
loss,W,X,y_target,train,learning_rate= Build_Graph()

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
one_hot_testMinstTarget=sess.run(tf.one_hot(testMinstTarget, 10) ) 
one_hot_validMinstTarget=sess.run(tf.one_hot(validMinstTarget, 10) )
one_hot_trainMinstTarget=sess.run(tf.one_hot(trainMinstTarget, 10) )

N=len(trainMinstData)
trainMinstData = np.reshape(trainMinstData, [N, 28*28])
validMinstData = np.reshape(validMinstData, [len(validMinstData), 28*28])
testMinstData = np.reshape(testMinstData, [len(testMinstData), 28*28])


iterations = 20000
num_epochs = int(iterations/30) #15000 Train Sample
tempresult=[]
result=[]
learn_rate=[0.005,0.001,0.0001]
for learn in learn_rate:
    for step in range(0,num_epochs):
            for i in range(0,30):
                start_index = i* 500
                minix=trainMinstData[start_index:start_index+500]
                miniy=one_hot_trainMinstTarget[start_index:start_index+500]
                y,err,train_r=sess.run([y_target,loss,train],feed_dict={X:minix,y_target:miniy,learning_rate:learn})
                #print(err)
                #_, err, currentW, currentb, yhat = sess.run([train, cross_entropy_loss, w, b, y_predicted], feed_dict={x: minix, y_target: miniy,learn_rate:learnrate, weight_decay:weightdecay})

            tempresult.append(err)
    result.append(tempresult)
    sess.run(init)


    
epochs=(len(tempresult))

x = np.arange(epochs)


color=['r','g','b']
for idx, val in enumerate(learn_rate):
    line, = plt.plot(x, result[idx],color=color[idx], label="learning rate: "+str(val))
    
plt.legend(loc='upper right', shadow=True, fontsize='x-large')

    
plt.ylabel('Training loss')
plt.xlabel('Number of Epochs')
plt.show()


#print(one_hot_trainMinstTarget.shape)

