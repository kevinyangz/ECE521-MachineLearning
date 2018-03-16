import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def Load_FaceData(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255
    data = np.reshape(data, [-1, 32*32])
    target = np.load(target_path)
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
    data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
    data[rnd_idx[trBatch + validBatch+1:-1],:]
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
    target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
    target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData,trainTarget,validData,validTarget,testData,testTarget

def Build_Graph():
    
    W = tf.Variable(tf.truncated_normal(shape=[1024,6], stddev=0.1), name='weights') #not too sure about stddev
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 1024], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,6], name='target_y')
    learn_rate=tf.placeholder(tf.float32,shape=[],name='learn_rate')
    weight_decay = tf.placeholder(tf.float32,shape=[],name='weight_decay')
    # Graph definition
    y_predicted = tf.matmul(X,W) + b
    crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target,logits=y_predicted))+ tf.divide(weight_decay,2)*tf.nn.l2_loss(W)
    optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)
    train = optimizer.minimize(loss=crossEntropyLoss)
    return crossEntropyLoss,W,b,X,y_target,train,learn_rate,y_predicted,weight_decay

loss,W,B,X,y_target,train,learning_rate,y_predicted_label,weight_decay= Build_Graph()
trainFaceData,trainFaceTarget,validFaceData,validFaceTarget,testFaceData,testFaceTarget= Load_FaceData("data.npy","target.npy",0)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

one_hot_testFaceTarget=sess.run(tf.one_hot(testFaceTarget, 6) )  #93
one_hot_validFaceTarget=sess.run(tf.one_hot(validFaceTarget, 6) ) #92
one_hot_trainFaceTarget=sess.run(tf.one_hot(trainFaceTarget, 6) ) #747

N=len(trainFaceData)
trainFaceData = np.reshape(trainFaceData, [N, 32*32])
validFaceData = np.reshape(validFaceData, [len(validFaceData), 32*32])
testFaceData = np.reshape(testFaceData, [len(testFaceData), 32*32])

iterations = 5000
batch_size = 300

num_batch_per_epoch = int(747/batch_size)
num_epochs = int(iterations/num_batch_per_epoch)

result=[]
learn=0.0001
weight=0.001 #best learn rate and weight accuracy
best_Y_Predicted=[]
valid_accuracy_result = []
valid_error_result = []
train_accuracy_result = []
train_error_result = []
tempresult = []
start_index = 0
prev_count = 0
for i in range(0,iterations):
    end_index = start_index + batch_size
    if(end_index<747):
        minix=trainFaceData[start_index:end_index]
        miniy=one_hot_trainFaceTarget[start_index:end_index]
        start_index = start_index + batch_size
    else:
        num_remaining = 747-start_index
        minix[0:num_remaining] = trainFaceData[start_index:]
        miniy[0:num_remaining] = one_hot_trainFaceTarget[start_index:]
        num_still_need = batch_size-num_remaining
        minix[num_remaining:batch_size] = trainFaceData[0:num_still_need]
        miniy[num_remaining:batch_size] = one_hot_trainFaceTarget[0:num_still_need]
        start_index = num_still_need
    err,train_r,weight_w,bias,y_predicted=sess.run([loss,train,W,B,y_predicted_label],feed_dict={X:minix,y_target:miniy,learning_rate:learn,weight_decay:weight})

    if(int((i*batch_size)/747) > prev_count):
        prev_count = int((i*batch_size)/747)
        #training data accuracy and error calculation
        y_predic =sess.run(tf.nn.softmax(sess.run(y_predicted_label,feed_dict={X:trainFaceData})))
        prediction_accuracy=tf.equal(tf.argmax(y_predic,1),tf.argmax(one_hot_trainFaceTarget,1))
        accur=sess.run(tf.reduce_mean(tf.cast(prediction_accuracy,tf.float32)))
        train_accuracy_result.append(accur)
        train_error_result.append(err)

        #calcuate the accuracy of valid data for each epoch
        y_predic =sess.run(tf.nn.softmax(sess.run(y_predicted_label,feed_dict={X:validFaceData})))
        prediction_accuracy=tf.equal(tf.argmax(y_predic,1),tf.argmax(one_hot_validFaceTarget,1))
        accur=sess.run(tf.reduce_mean(tf.cast(prediction_accuracy,tf.float32)))
        valid_accuracy_result.append(accur)
        err,y_predicted=sess.run([loss,y_predicted_label],feed_dict={X:validFaceData,y_target:one_hot_validFaceTarget,weight_decay:weight})
        valid_error_result.append(err)
        print(i, err)

y_predic =sess.run(tf.nn.softmax(sess.run(y_predicted_label,feed_dict={X:testFaceData})))
prediction_accuracy=tf.equal(tf.argmax(y_predic,1),tf.argmax(one_hot_testFaceTarget,1))
accur=sess.run(tf.reduce_mean(tf.cast(prediction_accuracy,tf.float32)))
#print("accuracy for test data is : " + str(accur))
sess.run(init)

x = np.arange(len(valid_accuracy_result))


plt.interactive(False)
# plot the cross entropy loss
line_valid_error = plt.plot(x, valid_error_result,color='r', label="Valid Data Set Loss")
line_test_accuracy = plt.plot(x, train_error_result,color='b', label="Train Data Set Loss")   
plt.legend(loc='best', shadow=True, fontsize='small')    
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Number of Epochs')
plt.show()

# plot the validation accuracy 
line_valid_accuracy = plt.plot(x, valid_accuracy_result,color='r', label="Valid Data Set Accuracy")
line_test_accuracy = plt.plot(x, train_accuracy_result,color='g', label="Test Data Set Accuracy")
plt.legend(loc='best', shadow=True, fontsize='small')    
plt.ylabel('Classification Accuracy')
plt.xlabel('Number of Epochs')
plt.show()




