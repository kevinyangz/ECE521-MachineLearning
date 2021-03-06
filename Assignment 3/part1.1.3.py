import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
    
    shape=input_tensor.get_shape().as_list()[1]
    #print(input_tensor.shape[1].value)
    initializer = tf.contrib.layers.xavier_initializer()
    W = tf.Variable(initializer([shape,n]),name='weights')
    b = tf.Variable(tf.zeros([1,n]), name='biases')
    output=tf.add(tf.matmul(input_tensor,W),b)
    #print(output.shape)
    return W,b,output

def build_graph():

    X = tf.placeholder(tf.float32, [None, 28*28], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')
    y_onehot = tf.one_hot(tf.to_int32(y_target), 10, 1.0, 0.0, axis = -1)
    learn_rate=tf.placeholder(tf.float32,shape=[],name='learn_rate')
    W0,b0,output = layer_block(X,1000)

    W1,b1,relu_out= layer_block(tf.nn.relu(output),10)

    y_predicted=tf.nn.softmax(relu_out)

    #print(y_predicted.shape)
    crossEntropyError = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
    labels = y_onehot, logits = relu_out),name='mean_cross_entropy')
   
    Wn=tf.reshape(W0,[1,-1])
    Wb=tf.reshape(W1,[1,-1])

    all_weight=tf.concat([Wn,Wb],1)

    weight_decay=tf.divide(0.0003,2)*tf.reduce_sum(all_weight*all_weight)
    #test=tf.equal(tf.argmax(y_predicted,1),tf.to_int64(y_target))
    #print(test.shape)
    accuracy = 1-(tf.reduce_mean(tf.to_float(tf.equal(tf.reshape(tf.argmax(y_predicted,1),[-1,1]),tf.to_int64(y_target)))))

    loss = crossEntropyError+weight_decay 
    # Training mechanism
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)
    train = optimizer.minimize(loss=loss)
    
    return W0,b0,W1,b1,X, y_target, y_predicted, crossEntropyError, train, accuracy,learn_rate    




trainMinstData,trainMinstTarget,validMinstData,validMinstTarget,testMinstData,testMinstTarget=load_data()
W0,b0,W1,b1,X,y_target, y_predicted_label, crossEntropyError, train, accuracy,learnrate= build_graph()
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
iterations = 20000
num_epochs = int(iterations/30) #15000 Train Sample
Trainresult=[]
TrainAcc=[]
TestAcc=[]
ValidAcc=[]

#learn_rate=[0.005,0.001,0.0001]
learn_rate=[0.001]
prev_training_loss=1
epoch_count=0
early_stop=0
epoch_count_valid=0
early_stop_valid = 0
valid_err =0
prev_valid_err=1

for learn in learn_rate:
    tempTrainresult=[]
    tempTrainacc=[]
    tempTestacc=[]
    tempValidacc=[]
    early_stop=0

    for step in range(0,num_epochs):
        for i in range(0,30):
            start_index = i* 500
            minix=trainMinstData[start_index:start_index+500]
            miniy=trainMinstTarget[start_index:start_index+500]   
            err,train_r,w0,bb0,w1,bb1,acc=sess.run([crossEntropyError,train,W0,b0,W1,b1,accuracy],feed_dict={X:minix,y_target:miniy,learnrate:learn})
    	
        tempTrainresult.append(err)#Record Training loss each epoch
	
	#figure 1 early stop
	if(err>=prev_training_loss and early_stop==0):
		epoch_count=epoch_count+1
	else:
		epoch_count=0;
	if(epoch_count == 5):
		print("figure 1 early stop found at epoch %s for learn rate %s"%(step,learn))
		early_stop=step

	prev_training_loss = err

	#figure 2 early stop
	valid_err = sess.run(accuracy,feed_dict={X:validMinstData,y_target:validMinstTarget})
	if(valid_err>=prev_valid_err and early_stop_valid==0):
		epoch_count_valid=epoch_count_valid+1
	else:
		epoch_count_valid=0;
	if(epoch_count_valid == 5):
		print("figure 2 early stop found at epoch %s for learn rate %s"%(step,learn))
		print("early stop test Acc: %s"%(sess.run(accuracy,feed_dict={X:testMinstData,y_target:testMinstTarget})))
		print("early stop train Acc: %s"%(sess.run(accuracy,feed_dict={X:trainMinstData,y_target:trainMinstTarget})))
		print("early stop valid Acc: %s"%(valid_err))
		early_stop_valid=step

	prev_valid_err = valid_err

        tempTestacc.append(sess.run(accuracy,feed_dict={X:testMinstData,y_target:testMinstTarget})) #Test Acc
        tempTrainacc.append(acc)#Train acc per mini-batch #Double check whether need the entire training set or not
        tempValidacc.append(sess.run(accuracy,feed_dict={X:validMinstData,y_target:validMinstTarget})) #Test Acc
        #print(str(step)+"-------"+str(bb0.shape)+"---"+str(bb1.shape))
		
    Trainresult.append(tempTrainresult)
    TrainAcc.append(tempTrainacc)
    TestAcc.append(tempTestacc)
    ValidAcc.append(tempValidacc) 
    		
		
	
    loss_result=sess.run(crossEntropyError,feed_dict={X:trainMinstData,y_target:trainMinstTarget})
    train_acc=sess.run([accuracy],feed_dict={X:trainMinstData,y_target:trainMinstTarget})
    test_acc=sess.run([accuracy],feed_dict={X:testMinstData,y_target:testMinstTarget})
    valid_acc,y_predicted=sess.run([accuracy,y_predicted_label],feed_dict={X:validMinstData,y_target:validMinstTarget})
    test=tf.reduce_mean(tf.to_float(tf.equal(tf.reshape(tf.argmax(y_predicted,1),[-1,1]),tf.to_int64(validMinstTarget))))
    

    print(sess.run(test))
    sess.run(init)


    print("learning rate: %s loss is %s"%(learn,loss_result)+" Train acc is %s"%train_acc +"Test acc is %s"%test_acc+"Valid acc is %s"%valid_acc)
#drawoption=1
#if(drawoption):
x=np.arange(num_epochs)
color=['r','g','b']
for idx,val in enumerate(learn_rate):
    #plt.figure(idx)
    plt.plot(x,Trainresult[idx],color='b',label="learn rate %s"%val)
    plt.axvline(x=early_stop,color='r',label="early stop at ephoch %s"%early_stop)
    #plt.plot(x,TrainAcc[idx],color=color[0],label="Train data error")
    #plt.plot(x,TestAcc[idx],color=color[1],label= "Test data error")
    #plt.plot(x,ValidAcc[idx],color=color[2],label="Valid data error")

    #plt.legend(loc='upper right',shadow=True,fontsize='x-large')
    #plt.ylabel('cross entropy loss')
    #plt.ylabel('error')
    #plt.xlabel('Number of epoches')
    #plt.title('Neural Network error vs number of epoches when learn rate '+str(val))
    #plt.show()
plt.legend(loc='upper right',shadow=True,fontsize='x-large')
plt.ylabel('cross entropy loss')
plt.xlabel('Number of epoches')
plt.title('Neural Network loss vs number of epoches')
plt.show()
    #x=np.arange(num_epochs)
    #for idx,val in enumerate(learn_rate):

#else:
x=np.arange(num_epochs)
color=['r','g','b','c']
for idx,val in enumerate(learn_rate):
    plt.figure(idx)

    plt.plot(x,TrainAcc[idx],color='g',label="Train data error")
    plt.plot(x,TestAcc[idx],color='b',label= "Test data error")
    plt.plot(x,ValidAcc[idx],color='c',label="Valid data error")

    plt.axvline(x=early_stop_valid,color='r',label="early stop at ephoch %s"%early_stop_valid)

    plt.legend(loc='upper right',shadow=True,fontsize='x-large')

    plt.ylabel('error')
    plt.xlabel('Number of epoches')
    plt.title('Neural Network error vs number of epoches when learn rate '+str(val))
    plt.show()
	
