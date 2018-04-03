import tensorflow as tf
import numpy as np


def load_data():
    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        Data=np.reshape(Data, [-1, Data.shape[1]*Data.shape[1]])
        Target=np.reshape(Target,[-1,1])
        Data=Data.astype(np.float32)
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
        return trainData,trainTarget,validData,validTarget,testData,testTarget

def layer_block(input_tensor,n):
    
    shape=input_tensor.get_shape().as_list()[0]
    initializer = tf.contrib.layers.xavier_initializer()
    W = tf.Variable(initializer([shape,n]),name='weights')
    b = tf.Variable(0.0, name='biases')
    output=tf.add(tf.matmul(tf.transpose(W),input_tensor),b)
    return W,b,output

def build_graph():

    X = tf.placeholder(tf.float32, [None, 28*28], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')
    y_onehot = tf.one_hot(tf.to_int32(y_target), 10, 1.0, 0.0, axis = -1)

    W0,b0,output = layer_block(tf.transpose(X),1000)
    W1,b1,relu_out= layer_block(tf.nn.relu(output),10)
    relu_out=tf.transpose(relu_out)
    y_predicted=tf.nn.softmax(relu_out)

    
    crossEntropyError = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
    labels = y_onehot, logits = relu_out),name='mean_cross_entropy')
    
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(y_predicted, -1),tf.to_int64(y_target))))
    loss = crossEntropyError 
    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.005)
    train = optimizer.minimize(loss=loss)
    
    return W0,b0,W1,b1,X, y_target, y_predicted, crossEntropyError, train, accuracy    



trainMinstData,trainMinstTarget,validMinstData,validMinstTarget,testMinstData,testMinstTarget=load_data()
W0,b0,W1,b1,X,y_target, y_predicted_label, crossEntropyError, train, accuracy= build_graph()
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
iterations = 20000
num_epochs = int(iterations/30) #15000 Train Sample
result=[]
accuracy_result=[]
learn_rate=[0.005]#,0.001,0.0001]
best_Y_Predicted=[]
print(trainMinstTarget.shape)
for learn in learn_rate:
    tempresult=[]
    tempacc=[]
    weight=[]
    bias=[]
    for step in range(0,num_epochs):
        for i in range(0,30):
            start_index = i* 500
            minix=trainMinstData[start_index:start_index+500]
            miniy=trainMinstTarget[start_index:start_index+500]   
            err,train_r,w0,bb0,w1,bb1,acc=sess.run([crossEntropyError,train,W0,b0,W1,b1,accuracy],feed_dict={X:minix,y_target:miniy})            
            tempresult.append(err)
            tempacc.append(acc)
        print(str(step)+"-------"+str(err)+"---"+str(acc))


    result.append(tempresult)
    accuracy_result.append(tempacc)
    loss_result=sess.run(crossEntropyError,feed_dict={X:trainMinstData,y_target:trainMinstTarget})
    loss_acc=sess.run(accuracy,feed_dict={X:trainMinstData,y_target:trainMinstTarget})

    print("learning rate: %s loss is %s"%(learn,loss_result))
    print("learning rate: %s acc is %s"%(learn,loss_acc))


