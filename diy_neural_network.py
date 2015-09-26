import numpy as np
from numpy import genfromtxt
from collections import Counter

# commercial = 0
# show = 1
# my_data[1:] excludes the header

def get_data_set(path):
	my_data= genfromtxt(path, delimiter=',')
	# returns [input,output]
	l = []
	for a in my_data[1:,-1:]:
		l += [a]
	l = np.array(l).T
	l = np.array(l).T
	return [my_data[1:,:-1], l]
# sigmoid function
def nonlin(x,deriv=False):

	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

def truthiness(a):
	return a == True
# basically if things are over .5 it's a 1, if it's less then it's a 0
def normalize_prediction(a):
	if a > .5:
		return 1.0
	else:
		return 0.0

# Training returns the list of weights that worked well.
def train(data, weights, iters = 1):
	weight = weights
	# number of steps
	for iter in xrange(iters):

	# forward propagation
		l0 = data[0]
		l1 = nonlin(np.dot(l0,weights))
	 
		# how much did we miss?
		l1_error = data[1] - l1

		# multiply how much we missed by the
		# slope of the sigmoid at the values in l1
		l1_delta = l1_error * nonlin(l1,True)
		
		# update weights
		# here is where we would use our random search algos
		weight += np.dot(l0.T,l1_delta)

	print "Output After Training:"
	return weight
	

def test(data, weights):
	weight = weights
	# number of steps
	# forward propagation
	l0 = data[0]
	l1 = nonlin(np.dot(l0,weights))
	result = map(normalize_prediction, l1)
	#rate = result == data[1]
	rate = data[1].flatten() == result
	correctness = len(filter(truthiness,rate)) / float(len(rate))
	# return predicted value, returns expected value
	return [result, data[1], correctness]

	

# [input,output]
[X,y] = get_data_set('C:/Users/tyson/Downloads/Curious_George Training and Test/Curious_George Training and Test/6_attributes_csv/Curious_George_train_features_10_percent.csv')
[I,O] = get_data_set('C:/Users/tyson/Downloads/Curious_George Training and Test/Curious_George Training and Test/6_attributes_csv/Curious_George_test_features.csv')

# seed random numbers to make calculation

# deterministic (just a good practice)

np.random.seed(1)

# initialize weights randomly with mean 0

syn0 = 2*np.random.random((5,1)) - 1

# These are the weights, this needs to be translated to some binary form likely.
#syn0 = []
#syn0 += [[1.0] * 168]
#syn0 += [[1.0] * 168]
#syn0 += [[1.0] * 168]
#syn0 += [[1.0] * 168]
#syn0 += [[1.0] * 168]

best_weights_ever = train([X,y], weights=syn0, iters=5000)

print test([I[1400],O[1]], weights=best_weights_ever)
