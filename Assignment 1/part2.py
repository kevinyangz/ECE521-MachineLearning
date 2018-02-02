import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from part1 import euclideanDistanceMultipleDimension


def findResponsibility(trainMatrix,testMatrix,k):
        distance = euclideanDistanceMultipleDimension(trainMatrix,testMatrix)
        distanceVector = tf.transpose(distance)
        distanceShape = tf.shape(distance)
        smallestIndices = tf.nn.top_k(-distanceVector, k).indices
        hotMatrix = tf.one_hot(smallestIndices, depth=distanceShape[0], on_value = 1/k)
        responsibilityMatrix = tf.transpose(tf.reduce_max(hotMatrix,1))
        responsibilityMatrix = tf.cast(responsibilityMatrix, tf.float64)
        return responsibilityMatrix








