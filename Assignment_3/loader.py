import csv
import numpy as np
class Loader:
    def makefloat(self,a):
        #print a
        return float(a)
    def makestr(self,a):
        return str(a)   
        
    def load_data(self,path):
        data = []
        results = []
        with open(path, 'rb') as csvfile:
             spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
             for row in spamreader:
                temp = map(self.makefloat, row)
                data.append(temp[:-1])
                results.append(temp[-1])    

        return [np.asarray(data), np.asarray(results)]
        
    def save_data(self,path, save):
<<<<<<< HEAD
        data = save[0].tolist()
        results = save[1].tolist()
        f = open(path, 'w')
        for i in range(len(data)):
                temp = str(data[i]).strip('[').strip(']').strip().replace("\s+", ',')
                temp = temp + ',' + str(int(results[i]))
                f.write(temp)
                f.write('\n')            

        return 
=======
        data = list(save[0])
        results = list(save[1])
        f = open(path, 'w')
        for i in range(len(data)):
                temp = str(data)
                temp + ',' + str(int(results[i]))
                f.write(temp)
                f.write('\n')            

        return [np.asarray(data), np.asarray(results)]
>>>>>>> a4d491099709dd7fd263850cba606962f9f001b2
