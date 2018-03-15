import tensorflow as tf
import numpy as np


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


init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)


trainFaceData,trainFaceTarget,validFaceData,validFaceTarget,testFaceData,testFaceTarget= Load_FaceData("data.npy","target.npy",0)
one_hot_testFaceTarget=sess.run(tf.one_hot(testFaceTarget, 6) ) 
one_hot_validFaceTarget=sess.run(tf.one_hot(validFaceTarget, 6) ) 
one_hot_trainFaceTarget=sess.run(tf.one_hot(trainFaceTarget, 6) ) 