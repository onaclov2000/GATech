from datetime import datetime
from SimpleCV import *
class Trainer():

    def __init__(self,classes, trainPaths, testPaths):
        self.classes = classes
        self.trainPaths = trainPaths
        self.testPaths = testPaths


    def getExtractors(self):
        hhfe = HueHistogramFeatureExtractor(5) #10
        ehfe = EdgeHistogramFeatureExtractor(5) #10
        haarfe = HaarLikeFeatureExtractor(fname='./SimpleCV/SimpleCV/Features/haar.txt')
        return [hhfe,ehfe,haarfe]

    def getClassifiers(self,extractors):
        mSVMProperties = {
        'KernelType':'RBF', #default is a RBF Kernel
        'SVMType':'C',     #default is C 
        'nu':None,          # NU for SVM NU
        'c':None,           #C for SVM C - the slack variable
        'degree':None,      #degree for poly kernels - defaults to 3
        'coef':None,        #coef for Poly/Sigmoid defaults to 0
        'gamma':None,       #kernel param for poly/rbf/sigma - default is 1/#samples       
        }
        # There are a bunch of options to try out.
        tree = TreeClassifier(extractors)
        bagged = TreeClassifier(extractors, flavor='Bagged')
        boosted = TreeClassifier(extractors, flavor='Boosted')
        forest = TreeClassifier(extractors, flavor='Forest')
        return [tree, bagged, boosted, forest]

    def train(self):
        self.classifiers = self.getClassifiers(self.getExtractors())
        
        print 'training'
        print len(self.classifiers)
        count = 0
        for classifier in self.classifiers:
            print str(datetime.now()) + " Start-- " + str(classifier)
            classifier.train(self.trainPaths,self.classes,verbose=True, savedata="train_results_" + str(type(classifier).__name__) + datetime.now().strftime("%Y%m%d-%H%M%S") + ".tab")
            print str(datetime.now()) + " End-- " + str(classifier)
            count = count + 1
            print classifier.test(self.testPaths,self.classes,verbose=False)

    def test(self,testPaths):
        for classifier in self.classifiers:
            print classifier.test(testPaths,self.classes,verbose=False)

    def visualizeResults(self,classifier,imgs):
        for img in imgs:
            className = classifier.classify(img)
            img.drawText(className,10,10,fontsize=60,color=Color.BLUE)         
        imgs.show()



def main():
    classes = ['show','commercial',]
    trainPaths = ['./videofolder/train/show','./videofolder/train/commercial' ]
    testPaths = ['./videofolder/test/show','./videofolder/test/commercial']
    print "main"
    trainer = Trainer(classes,trainPaths, testPaths)
    print "Class Made"
    trainer.train()

    imgs = ImageSet()

    for p in testPaths:
        imgs += ImageSet(p)
    random.shuffle(imgs)

    print "Result test"
    trainer.test(testPaths)

    #trainer.visualizeResults(tree,imgs)
    
main()
