import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from random import random

# thanks to this site for the basis of the code
#http://iamtrask.github.io/2015/07/12/basic-python-network/
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
	 
		# The remainder of this function (based on the original implementation) can/should be left unused.
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
	l1 = nonlin(np.dot(l0,weight))
	result = map(normalize_prediction, l1)
	#rate = result == data[1]
	rate = data[1].flatten() == result
	correctness = len(filter(truthiness,rate)) / float(len(rate))
	# return predicted value, returns expected value
	return [result, data[1], correctness]

# weight training will return a correctness value, that's all we're doing.
# in reality this is "test" without the other two datapoints.
# I just thought calling it weight training was kinda fun, like we're at the gym
# which I actually think is really boring, unlike this
def weight_training(data, training):
	return test(data, training)[2]

def randomized_hill_climb(func, data, targetScore, initialPoint=[0,0,0,0,0], initialStepSizes=[1,1,1,1,1], someAcceleration=1.2):
#from https://en.wikipedia.org/wiki/Hill_climbing	
#Continuous Space Hill Climbing Algorithm
   restarts = 0
   steps = 0
   currentPoint = 2*np.random.random((5,1)) - 1    # the zero-magnitude vector is common
   print "Setting Initial Weights"
   print currentPoint
   stepSize = initialStepSizes      # a vector of all 1's is common
   acceleration = someAcceleration # a value such as 1.2 is common
   candidate = [-acceleration, -1 / acceleration, 0, 1 / acceleration, acceleration]
   epsilon = .001
   while(1):
      steps = steps + 1
      before = func(data, currentPoint)
      for i in range(len(currentPoint)):
         best = -1
         bestScore = 0.0
         for j in range(4):         # try each of 5 candidate locations
            currentPoint[i][0] = currentPoint[i][0] + stepSize[i] * candidate[j]
            temp = func(data, currentPoint)
            currentPoint[i][0] = currentPoint[i][0] - stepSize[i] * candidate[j]
            if(temp > bestScore):
                 bestScore = temp
                 best = j
         if candidate[best] is not 0:
            currentPoint[i][0] = currentPoint[i][0] + stepSize[i] * candidate[best]
            stepSize[i] = stepSize[i] * candidate[best] # accelerate
      if (func(data, currentPoint) - before) < epsilon: 
         if (bestScore < targetScore):
            currentPoint = 2*np.random.random((5,1)) - 1 
            restarts = restarts + 1
            print "Selecting new random weights, last one only got us so far time for a random walk"			
            print currentPoint
         else:
            return [currentPoint, bestScore, restarts, steps]
      else:
         print [currentPoint, "bestScore: " + str(bestScore)]
		 
		 
def acceptance_probability():
   print "1"
   
def anneal(func, data, ):
    old_cost = func(sol)
    T = 1.0
    T_min = 0.00001
    alpha = 0.9
    while T > T_min:
        i = 1
        while i <= 100:
            new_sol = neighbor(sol)
            new_cost = func(new_sol)
            ap = acceptance_probability(old_cost, new_cost, T)
            if ap > random():
                sol = new_sol
                old_cost = new_cost
            i += 1
        T = T*alpha
    return sol, cost
	
# this is a random restart, no actual hill climb per se.
def randomized_restart(func, data):
	we = 2*np.random.random((5,1)) - 1
	# first opportunity
	result = func(data, we)
	old_result = 0.0
	count = 0
	best_result = 0.0
	result_list = []
	result_list.append(result)
	
	while (result < .875):
		old_result = 0.0
		# Here we're climbing a hill
		while (old_result < result):
			old_result = result
			we = we * 1.01
			result = func(data, we)
			result_list.append(result)
			count = count + 1
			if result >= .86:
				return [we,count,result]
		# I don't even know whats happening here, basically go the other way, not sure if there is value in this
		while (old_result > result):
			old_result = result
			we = we * .99
			result = func(data, we)
			result_list.append(result)
			count = count + 1
			if result >= .86:
				return [we,count,result]

		we = 2*np.random.random((5,1)) - 1
		result = func(data, we)
		result_list.append(result)
		count = count + 1
		if count > 20000:
			# this returns whatever was the last weight, 
			# and is likely not so good. not sure what to do other than 
			# write code to check for best result, and store weights as we go
			#
			return [we, count, result, result_list]
			
	return [we, count, result, result_list]
	
# [input,output]
[X,y] = get_data_set('C:/Users/tyson/Downloads/Curious_George Training and Test/Curious_George Training and Test/6_attributes_csv/Curious_George_train_features_10_percent.csv')
[I,O] = get_data_set('C:/Users/tyson/Downloads/Curious_George Training and Test/Curious_George Training and Test/6_attributes_csv/Curious_George_test_features.csv')

# seed random numbers to make calculation

# deterministic (just a good practice)

np.random.seed(1)

# initialize weights randomly with mean 0

syn0 = 2*np.random.random((5,1)) - 1

# Typically we'll go run through this train exercise and find a good weight, but
# instead we'll go ahead and just pick a set of weights, then try them against our training set
# if the results are good enough, then we're kinda done.
# so basically comment out the following line
# however I'd like to compare to the other options, so let's just let it go wild
#best_weights_ever = train([X,y], weights=syn0, iters=5000)
# and we do some weight training instead

rhc_best_weights_ever = randomized_hill_climb(weight_training, [X,y], .875)

#sa_best_weights_ever = simulated_annealing(weight_training, [X,y])
#ga_best_weights_ever = genetic_algorithm(weight_training, [X,y])

#print rhc_best_weights_ever[1] # number of iterations to find this result
#print rhc_best_weights_ever[2] # best result found (if we exit before count interrupts it should be at or over .85)

print test([I,O], weights=rhc_best_weights_ever[0])[2]
#print test([I[1400],O[1400]], weights=sa_best_weights_ever)
#print test([I[1400],O[1400]], weights=ga_best_weights_ever)
#print test([I[1400],O[1400]], weights=best_weights_ever)[2]
#print hc_best_weights_ever[3]
# plt.plot(hc_best_weights_ever[3])
# plt.ylabel('some numbers')
# plt.show()
