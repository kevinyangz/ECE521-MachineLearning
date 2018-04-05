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

    initializer = tf.contrib.layers.xavier_initializer()
    W = tf.Variable(initializer([shape,n]),name='weights')
    b = tf.Variable(tf.zeros(n), name='biases')
    output=tf.add(tf.matmul(input_tensor,W),b)

    return W,b,output

def build_graph():

    X = tf.placeholder(tf.float32, [None, 28*28], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')
    y_onehot = tf.one_hot(tf.to_int32(y_target), 10, 1.0, 0.0, axis = -1)
    learn_rate=tf.placeholder(tf.float32,shape=[],name='learn_rate')
    apply_dropout = tf.placeholder(tf.bool,name='apply_dropout')
    apply_weight_decay = tf.placeholder(tf.bool,name='apply_weight_decay')
    X_flatten=tf.reshape(X,[-1,28*28])
    W0,b0,output = layer_block(X_flatten,1000)
    activated_output = tf.nn.relu(output)   
    #apply the drop out here
    if (apply_dropout):
	print("dropout applied.")
	drop_out = tf.nn.dropout(activated_output, keep_prob =0.5)
	W1,b1,final_output= layer_block(drop_out,10)
    else: 
	W1,b1,final_output= layer_block(activated_output,10)



    #y_predicted=relu_out
    #print(y_predicted.shape)
    crossEntropyError = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_onehot, logits = final_output),name='mean_cross_entropy')
    y_predicted=tf.nn.softmax(final_output)
    Wn=tf.reshape(W0,[1,-1])
    Wb=tf.reshape(W1,[1,-1])

    all_weight=tf.concat([Wn,Wb],1)
    if(apply_weight_decay):
	print("weight decay applied")
	weight_decay=tf.divide(0,2)*tf.reduce_sum(all_weight*all_weight)
    else:
	weight_decay = 0
    #test=tf.equal(tf.argmax(y_predicted,1),tf.to_int64(y_target))
    #print(test.shape)
    accuracy = 1-(tf.reduce_mean(tf.to_float(tf.equal(tf.reshape(tf.argmax(y_predicted,1),[-1,1]),tf.to_int64(y_target)))))

    loss = crossEntropyError + weight_decay
    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)
    train = optimizer.minimize(loss=loss)
    
    return W0,b0,W1,b1,X, y_target, y_predicted, crossEntropyError, train, accuracy,learn_rate    


#parameters input
apply_dropout = 1
apply_weight_decay = 0

trainMinstData,trainMinstTarget,validMinstData,validMinstTarget,testMinstData,testMinstTarget=load_data()
W0,b0,W1,b1,X,y_target, y_predicted_label, crossEntropyError, train, accuracy,learnrate= build_graph()


init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
iterations = 1000
num_epochs = int(iterations/30) #15000 Train Sample
Trainresult=[]
TrainAcc=[]
TestAcc=[]
ValidAcc=[]

learn_rate=[0.001]

for learn in learn_rate:
    tempTrainresult=[]
    tempTrainacc=[]
    tempTestacc=[]
    tempValidacc=[]
    tempTrainresult_d=[]
    tempTrainacc_d=[]
    tempTestacc_d=[]
    tempValidacc_d=[]
    for step in range(0,num_epochs):
        for i in range(0,30):
            start_index = i* 500
            minix=trainMinstData[start_index:start_index+500]
            miniy=trainMinstTarget[start_index:start_index+500]   
            err,train_r,w0,bb0,w1,bb1,acc=sess.run([crossEntropyError,train,W0,b0,W1,b1,accuracy],feed_dict={X:minix,y_target:miniy,learnrate:learn,apply_dropout:False,apply_weight_decay:False})
	    err_d,train_r_d,w0_d,bb0_d,w1_d,bb1_d,acc_d=sess.run([crossEntropyError,train,W0,b0,W1,b1,accuracy],feed_dict={X:minix,y_target:miniy,learnrate:learn,apply_dropout:True,apply_weight_decay:False})
    	
        tempTrainresult.append(err)#Record Training loss each epoch
        tempTestacc.append(sess.run(accuracy,feed_dict={X:testMinstData,y_target:testMinstTarget})) #Test Acc
        tempTrainacc.append(acc)#Train acc per mini-batch #Double check whether need the entire training set or not
        tempValidacc.append(sess.run(accuracy,feed_dict={X:validMinstData,y_target:validMinstTarget})) #Test Acc
	#for drop out
        tempTrainresult_d.append(err_d)#Record Training loss each epoch
        tempTestacc_d.append(sess.run(accuracy_d,feed_dict={X:testMinstData,y_target:testMinstTarget})) #Test Acc
        tempTrainacc_d.append(acc_d)#Train acc per mini-batch #Double check whether need the entire training set or not
        tempValidacc.append(sess.run(accuracy_d,feed_dict={X:validMinstData,y_target:validMinstTarget})) #Test Acc
        #print(str(step)+"-------"+str(bb0.shape)+"---"+str(bb1.shape))
		
    Trainresult.append(tempTrainresult)
    TrainAcc.append(tempTrainacc)
    TestAcc.append(tempTestacc)
    ValidAcc.append(tempValidacc) 

    Trainresult_d.append(tempTrainresult_d)
    TrainAcc_d.append(tempTrainacc_d)
    TestAcc_d.append(tempTestacc_d)
    ValidAcc_d.append(tempValidacc_d)
    		
		
	
    loss_result=sess.run(crossEntropyError,feed_dict={X:trainMinstData,y_target:trainMinstTarget})
    train_acc=sess.run([accuracy],feed_dict={X:trainMinstData,y_target:trainMinstTarget})
    test_acc=sess.run([accuracy],feed_dict={X:testMinstData,y_target:testMinstTarget})
    valid_acc,y_predicted=sess.run([accuracy,y_predicted_label],feed_dict={X:validMinstData,y_target:validMinstTarget})
    #test=tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(y_predicted,-1),tf.to_int64(validMinstTarget))))
    

    #print(sess.run(test))
    sess.run(init)


    print("learning rate: %s loss is %s"%(learn,loss_result)+" Train error is %s"%train_acc +"Test error is %s"%test_acc+"Valid error is %s"%valid_acc)
drawoption=1
if( not drawoption):
	x=np.arange(num_epochs)
	color=['r','yellow','g','b','teal','violet']
	for idx,val in enumerate(learn_rate):
	    #plt.figure(idx)
	    plt.plot(x,Trainresult[idx],color=color[idx],label="learn rate %s"%val)
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
if(drawoption):
    
	x=np.arange(num_epochs)
	color=['r','g','b','yellow','teal','violet']
	for idx,val in enumerate(learn_rate):
	    plt.figure(idx)

	    plt.plot(x,TrainAcc[idx],color=color[0],label="Train data error")
	    #plt.plot(x,TestAcc[idx],color=color[1],label= "Test data error")
	    plt.plot(x,ValidAcc[idx],color=color[2],label="Valid data error")

	    plt.legend(loc='best',shadow=True,fontsize='x-large')

	    plt.ylabel('error')
	    plt.xlabel('Number of epochs')
	    plt.title('Neural Network Error vs Number of Epochs with Learn Rate '+str(val))
	    plt.show()
	
