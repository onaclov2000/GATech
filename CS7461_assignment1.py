from SimpleCV import *
class Trainer():

    def __init__(self,classes, trainPaths):
        self.classes = classes
        self.trainPaths = trainPaths


    def getExtractors(self):
        hhfe = HueHistogramFeatureExtractor(10)
        ehfe = EdgeHistogramFeatureExtractor(10)
        haarfe = HaarLikeFeatureExtractor(fname=None, do45=True)
        return [hhfe,ehfe]#,haarfe]

    def getClassifiers(self,extractors):
        svm = SVMClassifier(extractors)
        #svm = SVMClassifier(extractors, properties)
        # There are a bunch of options to try out.
        tree = TreeClassifier(extractors)
        bagged = TreeClassifier(extractors, flavor='Bagged')
        boosted = TreeClassifier(extractors, flavor='Boosted')
        forest = TreeClassifier(extractors, flavor='Forest')
        bayes = NaiveBayesClassifier(extractors)
        onenn = KNNClassifier(extractors)
        twonn = KNNClassifier(extractors, k=2)
        threenn = KNNClassifier(extractors, k=3)
        # Oddly the 3 KNN classifiers below have some kind of erro in the implementation I chose.
        #twonn_dist_Euclidean = KNNClassifier(extractors, k=2, dist='Euclidean')
        #twonn_dist_Manhattan = KNNClassifier(extractors, k=2, dist='Manhattan')
        #twonn_dist_Hamming = KNNClassifier(extractors, k=2, dist='Hamming')
        #return [twonn_dist_Hamming]
        return [svm,tree, bagged, boosted, forest,bayes,onenn, twonn, threenn]

    def train(self):
        self.classifiers = self.getClassifiers(self.getExtractors())
        print 'training'
        print len(self.classifiers)
        for classifier in self.classifiers:
            print '.'
            classifier.train(self.trainPaths,self.classes,verbose=False)

    def test(self,testPaths):
        for classifier in self.classifiers:
            print classifier.test(testPaths,self.classes,verbose=False)

    def visualizeResults(self,classifier,imgs):
        for img in imgs:
            className = classifier.classify(img)
            img.drawText(className,10,10,fontsize=60,color=Color.BLUE)         
        imgs.show()


classes = ['show','commercial',]

def main():
    trainPaths = ['/media/sf_machine_learning/videofolder/train/show','/media/sf_machine_learning/videofolder/train/commercial' ]
    testPaths = ['/media/sf_machine_learning/videofolder/test/show','/media/sf_machine_learning/videofolder/test/commercial']
    print "main"
    trainer = Trainer(classes,trainPaths)
    print "Class Made"
    trainer.train()
    #tree = trainer.classifiers[0]

    imgs = ImageSet()

    for p in testPaths:
        imgs += ImageSet(p)
    random.shuffle(imgs)

    print "Result test"
    trainer.test(testPaths)

    #trainer.visualizeResults(tree,imgs)

main()
