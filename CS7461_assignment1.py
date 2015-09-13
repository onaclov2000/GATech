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
        haarfe = HaarLikeFeatureExtractor(fname='/SimpleCV-develop/SimpleCV/Features/haar.txt')
        return [hhfe,ehfe]#,haarfe]

    def getClassifiers(self,extractors):
        classifier = SVMClassifier(extractors)
        return [classifier]

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
