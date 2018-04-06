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



def build_graph(weight_decay_factor,num_layer,drop_out,num_nodes):

    X = tf.placeholder(tf.float32, [None, 28*28], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')
    y_onehot = tf.one_hot(tf.to_int32(y_target), 10, 1.0, 0.0, axis = -1)
    learn_rate=tf.placeholder(tf.float32,shape=[],name='learn_rate')
    X_flatten=tf.reshape(X,[-1,28*28])
    input_to_nodes = X_flatten
    weight_decay = 0
    for i in range (0,num_layer):
        W0,b0,output = layer_block(input_to_nodes,num_nodes)
        activated_output = tf.nn.relu(output)   
		#apply the drop out 
        if(drop_out):
		    input_to_nodes = tf.nn.dropout(activated_output,keep_prob=0.5)
        else:   
		    input_to_nodes = activated_output
        weight_decay += weight_decay_factor*tf.reduce_sum(W0*W0)

    #output layer
    W_out,b_out,output_layer = layer_block(input_to_nodes,10)
    crossEntropyError = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_onehot, logits = output_layer),name='mean_cross_entropy')
    y_predicted=tf.nn.softmax(output_layer)
    accuracy = 1-(tf.reduce_mean(tf.to_float(tf.equal(tf.reshape(tf.argmax(y_predicted,1),[-1,1]),tf.to_int64(y_target)))))
    loss = crossEntropyError  + weight_decay
    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)
    train = optimizer.minimize(loss=loss)
    
    return W0,b0,W_out,b_out,X, y_target, y_predicted, crossEntropyError, train, accuracy,learn_rate


#parameters input

trainMinstData,trainMinstTarget,validMinstData,validMinstTarget,testMinstData,testMinstTarget=load_data()




iterations = 10000
num_epochs = int(iterations/30) #15000 Train Sample
Trainresult=[]
TrainAcc=[]
TestAcc=[]
ValidAcc=[]


learn=0.001
num_of_random = 5

for i in range(0,num_of_random):
    np.random.seed(i*1000430759)
    weight_decay_factor= 0.00019946 #np.exp(np.random.uniform(-9,-6)) #Random Natural Log for weight decay
    number_of_layer=4 # np.int64(np.random.uniform(1,5.5)) #the number of layers from 1 to 5
    learn_rate_factor=0.00259209#np.exp(np.random.uniform(-7.5,-4.5)) #Random Natural Log for learn rate
    dropout_flag= False #np.random.choice([True, False])
    number_of_nodes =168 # np.int64(np.random.uniform(100,500.5)) #the number of hidden units per layer
    print("Model " + str(i) + ": Weight Decay Factor of: " + str(weight_decay_factor) + " Number of Layer of: " + str(number_of_layer) +" Learning Rate of: " + str(learn_rate_factor) + " Droupout?: " +str(dropout_flag) + " Num of Nodes:" + str(number_of_nodes))

    W0,b0,W1,b1,X,y_target, y_predicted_label, crossEntropyError, train, accuracy,learn_rate= build_graph(weight_decay_factor,number_of_layer,dropout_flag,number_of_nodes)
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
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
            err,train_r,w0,bb0,w1,bb1,acc=sess.run([crossEntropyError,train,W0,b0,W1,b1,accuracy,],feed_dict={X:minix,y_target:miniy,learn_rate:learn_rate_factor})
        tempTrainresult.append(err)#Record Training loss each epoch
        tempTestacc.append(sess.run(accuracy,feed_dict={X:testMinstData,y_target:testMinstTarget})) #Test Acc
        tempTrainacc.append(acc)#Train acc per mini-batch #Double check whether need the entire training set or not
        tempValidacc.append(sess.run(accuracy,feed_dict={X:validMinstData,y_target:validMinstTarget})) #Test Acc
		
    Trainresult.append(tempTrainresult)
    TrainAcc.append(tempTrainacc)
    TestAcc.append(tempTestacc)
    ValidAcc.append(tempValidacc) 

		
	
    loss_result=sess.run(crossEntropyError,feed_dict={X:trainMinstData,y_target:trainMinstTarget})
    train_acc=sess.run([accuracy],feed_dict={X:trainMinstData,y_target:trainMinstTarget})
    test_acc=sess.run([accuracy],feed_dict={X:testMinstData,y_target:testMinstTarget})
    valid_acc,y_predicted=sess.run([accuracy,y_predicted_label],feed_dict={X:validMinstData,y_target:validMinstTarget})



    print("Train error is %s"%train_acc +"Test error is %s"%test_acc+"Valid error is %s"%valid_acc)
drawoption=1
if( drawoption):
	x=np.arange(num_epochs)
	color=['r','yellow','g','b','teal','violet']
	for idx in range(0,num_of_random):
	    plt.plot(x,Trainresult[idx],color=color[idx],label="Model %d" %idx)
	    plt.xlabel('Number of epochs')
	plt.legend(loc='best',shadow=True,fontsize='x-large')
	plt.ylabel('Cross Entropy Loss')
	plt.xlabel('Number of Epochs')
	plt.title('Cross Entropy Loss vs number of epoches for different models')
	plt.show()

if( drawoption):
	x=np.arange(num_epochs)
	color=['r','yellow','g','b','teal','violet']
	for idx in range(0,num_of_random):
	    plt.plot(x,TestAcc[idx],color=color[idx],label= "Model %d"%idx)
	    plt.xlabel('Number of epochs')
	plt.legend(loc='best',shadow=True,fontsize='x-large')
	plt.ylabel('Test Error')
	plt.xlabel('Number of Epochs')
	plt.title('Test Error vs number of Epoches for Different Models')
	plt.show()

if( drawoption):
	x=np.arange(num_epochs)
	color=['r','yellow','g','b','teal','violet']
	for idx in range(0,num_of_random):
	    plt.plot(x,ValidAcc[idx],color=color[idx],label= "Model %d"%idx)
	    plt.xlabel('Number of epochs')
	plt.legend(loc='best',shadow=True,fontsize='x-large')
	plt.ylabel('Valie Error')
	plt.xlabel('Number of Epochs')
	plt.title('Valid Error vs Number of Epoches for Different Models')
	plt.show()

if(not drawoption):
    x=np.arange(num_epochs)
    color=['r','g','b','yellow','teal','violet']
	#plt.figure(idx)
    plt.plot(x,TrainAcc[0],color=color[0],label="Train data error without Dropout")
    plt.plot(x,TrainAcc[1],color=color[1],label="Train data error with Dropout")
    plt.plot(x,ValidAcc[0],color=color[2],label="Valid data error without Dropout")
    plt.plot(x,ValidAcc[1],color=color[3],label="Valid data error with Dropout")

    plt.legend(loc='best',shadow=True,fontsize='x-large')

    plt.ylabel('error')
    plt.xlabel('Number of epochs')
    plt.title('Neural Network Error vs Number of Epochs with/without Dropout')
    plt.show()
	
