import random
import os
import re
import sys


def run_nn(path):
	print path	
	res = os.popen('java -classpath $CLASSPATH:/home/tyson/machine_learning/weka-3-6-12/weka.jar weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a -t ' + path)
	result_list = []
	for line in res:
	        result_list.append(line)
		if "Correctly Classified Instances" in line:
			goal = re.search('[0-9]+[ ]+([0-9]+(.[0-9]+)?)', line).group(1)
			print goal
			sys.stdout.flush()

	print ''.join(result_list[-8:])
	sys.stdout.flush()

import glob, os
os.chdir("datasets/")
for file in glob.glob("*.arff"):
    if 'live' not in file.lower():
	    run_nn(file)
