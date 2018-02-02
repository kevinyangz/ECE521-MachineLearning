import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from part1 import euclideanDistanceMultipleDimension



def findResponsibility(trainMatrix,testMatrix,k):
        distance = euclideanDistanceMultipleDimension(trainMatrix,testMatrix)
        distanceVector = tf.transpose(distance)

        smallestIndices = tf.nn.top_k(tf.negative(distanceVector), k).indices
        hotMatrix = tf.one_hot(smallestIndices, depth= tf.shape(distance)[0], on_value = 1/k)
        responsibilityMatrix = tf.transpose(tf.reduce_max(hotMatrix,1))
        responsibilityMatrix = tf.cast(responsibilityMatrix, tf.float64)
        return responsibilityMatrix

def KNNRegression(trainData,testData,trainTarget,testTarget,k):
  		
        responsibility = findResponsibility(trainData,testData,k)
        trainTarget = tf.transpose(trainTarget)    
        sess=tf.InteractiveSession()
        init=tf.global_variables_initializer()
        sess.run(init)

        y_hat = tf.matmul(trainTarget,responsibility)
        # Error definition
        #TF build in MSE mse=tf.losses.mean_squared_error(testTarget,tf.transpose(y_hat))
        meanSquaredError = tf.divide(tf.reduce_mean(tf.reduce_sum(tf.square(tf.transpose(y_hat) - testTarget), 
                                                        reduction_indices=1, 
                                                        name='squared_error'), 
                                          name='mean_squared_error'),2)
        

        testDatacopy=testData
        testTargetcopy=testTarget
        graph_test_data = tf.reshape(testDatacopy,[-1])
        graph_test_target = tf.reshape(testTargetcopy,[-1])
        print("MSE for k= %s"%(k))
        print(sess.run(meanSquaredError))

        lists = sorted(zip(*[sess.run(graph_test_data), sess.run(graph_test_target)]))
        new_x, new_y = list(zip(*lists))
        testDatacopy1=testTarget

        plt.style.use("ggplot")

        plt.figure()
        plt.scatter(sess.run(tf.reshape(testDatacopy,[-1])), sess.run(tf.reshape(tf.transpose(y_hat),[-1])), color='red')
        plt.plot(new_x, new_y, color='lightblue')
        red_patch = mpatches.Patch(color='lightblue', label='Real Result')
        lightblue_patch = mpatches.Patch(color='red', label='Predicted Result')
        plt.title("KNN regression Result for K =%s MSE is %s "%(k,sess.run(meanSquaredError)), fontsize=15)
                #print("Accuracy for the test set is: %s %s, for k = %s"%(accuracy,'%',k_best))
        
        plt.legend(handles=[red_patch,lightblue_patch])
        plt.show()
        plt.savefig("k=%s.png"%(k))

def main():

		sess = tf.InteractiveSession()
		init = tf.global_variables_initializer()
		sess.run(init)


		np.random.seed(521)
		Data = np.linspace(0.0 , 11.0 , num =1000) [:, np. newaxis]
		Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
		+ 0.5 * np.random.randn(1000 , 1)
		randIdx = np.arange(1000)
		np.random.shuffle(randIdx)
		trainData, trainTarget = Data[randIdx[:800]], Target[randIdx[:800]]
		validData, validTarget = Data[randIdx[800:900]], Target[randIdx[800:900]]
		testData, testTarget = Data[randIdx[900:1000]], Target[randIdx[900:1000]]

		klist=[1,3,5,50]
		for k in klist:
		        KNNRegression(trainData,testData,trainTarget,testTarget,k)

		print("validData")
		for k in klist:    
		       KNNRegression(trainData,validData,trainTarget,validTarget,k)


if __name__ == "__main__":
    main()

