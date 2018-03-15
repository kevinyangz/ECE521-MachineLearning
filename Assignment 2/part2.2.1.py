import tensorflow as tf
import numpy as np


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

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)



trainMinstData,trainMinstTarget,validMinstData,validMinstTarget,testMinstData,testMinstTarget= Load_NotMinst()
one_hot_testMinstTarget=sess.run(tf.one_hot(testMinstTarget, 10) ) 
one_hot_validMinstTarget=sess.run(tf.one_hot(validMinstTarget, 10) )
one_hot_trainMinstTarget=sess.run(tf.one_hot(trainMinstTarget, 10) )


