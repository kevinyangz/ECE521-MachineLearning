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
   
    Wn=tf.reshape(W0,[1,-1])
    Wb=tf.reshape(W1,[1,-1])
    print(W0)
    print(W1)
    print(Wn)
    print(Wb)
    all_weight=tf.concat([Wn,Wb],1)
    print(all_weight)
    weight_decay=tf.divide(0.0003,2)*tf.reduce_sum(all_weight*all_weight)
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.reshape(tf.argmax(y_predicted, 1),[-1,1]),tf.to_int64(y_target))))
    loss = crossEntropyError+weight_decay 
    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.005)
    train = optimizer.minimize(loss=loss)
    
    return weight_decay,W0,b0,W1,b1,X, y_target, y_predicted, crossEntropyError, train, accuracy    



trainMinstData,trainMinstTarget,validMinstData,validMinstTarget,testMinstData,testMinstTarget=load_data()
weight_deca,W0,b0,W1,b1,X,y_target, y_predicted_label, crossEntropyError, train, accuracy= build_graph()
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
iterations = 20000
num_epochs = int(iterations/30) #15000 Train Sample
Trainresult=[]
TrainAcc=[]
TestAcc=[]
ValidAcc=[]

learn_rate=[0.005,0.001,0.0001]

for learn in learn_rate:
    tempTrainresult=[]
    tempTrainacc=[]
    tempTestacc=[]
    tempValidacc=[]

    for step in range(0,num_epochs):
        for i in range(0,30):
            start_index = i* 500
            minix=trainMinstData[start_index:start_index+500]
            miniy=trainMinstTarget[start_index:start_index+500]   
            test,err,train_r,w0,bb0,w1,bb1,acc=sess.run([weight_deca,crossEntropyError,train,W0,b0,W1,b1,accuracy],feed_dict={X:minix,y_target:miniy})
		            
		#Loss result Collection per epoch
    tempTrainresult.append(err)#Record Training loss each epoch
		#Accuracy result per epoch
    tempTestacc.append(sess.run(accuracy,feed_dict={X:testMinstData,y_target:testMinstTarget})) #Test Acc
    tempTrainacc.append(acc)#Train acc per mini-batch #Double check whether need the entire training set or not
    tempValidacc.append(sess.run(accuracy,feed_dict={X:validMinstData,y_target:validMinstTarget})) #Test Acc
        #print(str(step)+"-------"+str(err)+"---"+str(acc))
		
    Trainresult.append(tempTrainresult)
    TrainAcc.append(tempTrainacc)
    TestAcc.append(tempTestacc)
    ValidAcc.append(tempValidacc)
			
		
	
    loss_result=sess.run(crossEntropyError,feed_dict={X:trainMinstData,y_target:trainMinstTarget})
    train_acc=sess.run([accuracy],feed_dict={X:trainMinstData,y_target:trainMinstTarget})
    test_acc=sess.run([accuracy],feed_dict={X:testMinstData,y_target:testMinstTarget})
    valid_acc=sess.run([accuracy],feed_dict={X:validMinstData,y_target:validMinstTarget})


    print("learning rate: %s loss is %s"%(learn,loss_result)+" Train acc is %s"%train_acc +"Test acc is %s"%test_acc+"Valid acc is %s"%valid_acc)


    #x=np.arange(num_epochs)
    #for idx,val in enumerate(learn_rate):
	



