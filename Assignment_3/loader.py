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
        data = list(save[0])
        results = list(save[1])
        f = open(path, 'w')
        for i in range(len(data)):
                temp = str(data)
                temp + ',' + str(int(results[i]))
                f.write(temp)
                f.write('\n')            

        return [np.asarray(data), np.asarray(results)]
