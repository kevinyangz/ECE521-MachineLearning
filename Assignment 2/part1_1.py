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

	#constants/ variable setup/definition
	mini_batch_size = 500
	iteration = 20000
	#Todo: change the hardcode 7
	epoch = iteration/7
	learning_rate = 0.005
	epoch_MSE =[]


	inputX = tf.placeholder(tf.float32)
	inputY = tf.placeholder(tf.float32)

	#hyper parameters to be found
	W = tf.Variable(rng.randn(), name="weight")
	b = tf.Variable(rng.randn(), name="bias")
	linear_pred = tf.add(tf.multiply(inputX, tf.transpose(W)), b)
	#define the loss function, with lambda = 0
	MSE = tf.reduce_mean(tf.reduce_sum((linear_pred - inputY)**2, 1))/2
	training_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(MSE)
	

	#initialize all the parameters
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)

	for e in epoch:
		sess.run(training_optimizer,feed_dict={inputX:trainData, inputY:trainTarget})
		MSE = sess.run(MSE,feed_dict={inputX:trainData,inputY:trainTarget})
		epoch_MSE.append(MSE)
		print (MSE)



if __name__ == "__main__":
    main()