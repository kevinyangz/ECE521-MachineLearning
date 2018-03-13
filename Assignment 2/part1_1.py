import tensorflow as tf
import numpy as np



def load_data():
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


def main():
	load_data();
	print ("finish loading data")
	trainX = tf.placeholder(tf.float32)
	trainY = tf.placeholder(tf.float32)
	inputX = tf.placeholder(tf.float32)
	inputY = tf.placeholder(tf.float32)


	W = tf.Variable(rng.randn(), name="weight")
	b = tf.Variable(rng.randn(), name="bias")
	pred = tf.add(tf.multiply(trainX, W), b)
	#define the loss function, with lambda = 0
	MSE = tf.reduce_sum((predY - inputY)**2, 1)/2*n_samples


if __name__ == "__main__":
    main()