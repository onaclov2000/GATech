#f = open('temp.txt')
#accuracy_score = []
#accuracy_score.append(f.readline())
import itertools

f = open('results_curious_random_projection_kmeans.txt', 'r')
file_text = f.readlines() 
accuracy_score = []
for line in range(len(file_text)):
    if "Accuracy Score" in file_text[line]:
	accuracy_score.append(file_text[line + 1].strip())

print "It took " + str(accuracy_score.index(max(accuracy_score))) + "iterations to find the max of 100 runs Curious George Dataset"
print max(accuracy_score)


f = open('results_live_random_projection_kmeans.txt', 'r')
file_text = f.readlines() 
accuracy_score = []
for line in range(len(file_text)):
    if "Accuracy Score" in file_text[line]:
	accuracy_score.append(file_text[line + 1].strip())

print "It took " + str(accuracy_score.index(max(accuracy_score))) + "iterations to find the max of 100 runs Live Dataset"
print max(accuracy_score)
