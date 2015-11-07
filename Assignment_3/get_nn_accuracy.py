
import itertools

f = open('nn_results.txt', 'r')
file_text = f.readlines() 
accuracy_score = []
for line in range(len(file_text)):
    if "arff" in file_text[line]:
	accuracy_score.append(file_text[line].strip() + ',' + file_text[line + 1].strip())


print '\n'.join(accuracy_score)
