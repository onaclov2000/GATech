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
        haarfe = HaarLikeFeatureExtractor(fname=None, do45=True)
        print hhfe.getFieldNames()
        print ehfe.getFieldNames()
        return [hhfe,ehfe]#,haarfe]

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
        nn = NeuralNetworkClassifier(extractors)
        #svm = SVMClassifier(extractors, properties=mSVMProperties)
        #svm = SVMClassifier(extractors, properties)
        # There are a bunch of options to try out.
        #tree = TreeClassifier(extractors)
        #bagged = TreeClassifier(extractors, flavor='Bagged')
        #boosted = TreeClassifier(extractors, flavor='Boosted')
        #forest = TreeClassifier(extractors, flavor='Forest')
        #bayes = NaiveBayesClassifier(extractors)
        #onenn = KNNClassifier(extractors)
        #twonn = KNNClassifier(extractors, k=2)
        #threenn = KNNClassifier(extractors, k=4)
        # Oddly the 3 KNN classifiers below have some kind of erro in the implementation I chose.
        #twonn_dist_Euclidean = KNNClassifier(extractors, k=2, dist='Euclidean')
        #twonn_dist_Manhattan = KNNClassifier(extractors, k=2, dist='Manhattan')
        #twonn_dist_Hamming = KNNClassifier(extractors, k=2, dist='Hamming')
        #return [twonn_dist_Hamming]
        #return [svm, tree]
        return [svm,tree, bagged, boosted, forest,bayes,onenn, twonn, threenn]

    def train(self):
        self.classifiers = self.getClassifiers(self.getExtractors())
        
        print 'training'
        print len(self.classifiers)
        count = 0
        for classifier in self.classifiers:
            print str(datetime.now()) + " Start-- " + str(classifier)
            classifier.train(self.trainPaths,self.classes,verbose=True, savedata="train_results_" + str(count) + ".txt")
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
    # path for video folder is on my server
    trainPaths = ['../videofolder/train/show','../videofolder/train/commercial' ]
    testPaths = ['../videofolder/test/show','../videofolder/test/commercial']
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
