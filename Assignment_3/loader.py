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
        data = save[0].tolist()
        results = save[1].tolist()
        f = open(path, 'w')
        for i in range(len(data)):
                temp = str(data[i]).strip('[').strip(']').strip().replace("\s+", ',')
                temp = temp + ',' + str(int(results[i]))
                f.write(temp)
                f.write('\n')            

        return 

