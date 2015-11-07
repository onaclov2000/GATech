import time
from loader import Loader
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn import datasets
from sklearn import random_projection
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
import itertools

def make_ellipses(gmm, ax):
    for n, color in enumerate('rg'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

# works for binary classification (A,B)
def print_results(data):
    # Split input data by row and then on spaces
    rows = [ line.strip().split(' ') for line in data.split('\n') ]

    column_second = max(len(rows[1][0]), len(rows[2][0]))
    print rows[0][0] + " " * column_second + rows[0][1]
    print "-" * len(rows[2][0] + " " + " " * (len(rows[1][0]) - len(rows[2][0])) + rows[2][1])
    if column_second == len(rows[1][0]):
        print rows[1][0] + " " + rows[1][1] + " " + rows[0][0] + " " + str(float(rows[1][0])/(float(rows[1][1]) + float(rows[1][0])))
        print rows[2][0] + " " + " " * (len(rows[1][0]) - len(rows[2][0])) + rows[2][1] + " " + rows[0][1] + " " + str(float(rows[2][1])/(float(rows[2][1]) + float(rows[2][0])))
    else:
        print rows[1][0] + " " + " " * (len(rows[1][0]) - len(rows[2][0])) + rows[1][1] + " " + rows[0][0] + " " + str(float(rows[1][0])/(float(rows[1][1]) + float(rows[1][0])))
        print rows[2][0] + " " + rows[2][1]  + " " + rows[0][1] + " " + str(float(rows[2][1])/(float(rows[2][1]) + float(rows[2][0])))

def print_confusion_matrix(title, Y_kmeans, y):
    correct = 0
    incorrect = 0
    false_positive = 0
    true_positive = 0
    print len(Y_kmeans)
    print len(y)
    for i in range(len(Y_kmeans)):
        if Y_kmeans[i] == y[i]:
            correct += 1
            if y[i] == 0:
                true_positive += 1
        else:
            if y[i] == 0:
                false_positive += 1
            incorrect += 1

    print title + " K Means Confusion Matrix"
    result = ""
    result += "a b\n" 
    result += str(true_positive) + " " + str(incorrect - false_positive) + " a \n"
    result += str(false_positive) + " " +str(correct - true_positive) +  " b\n"
    print_results (result)
    print "Accuracy: " + str(correct/float((correct + incorrect)))


def split_data(a,i,n):
    x = []
    y = []
    for element in a:
        x.append(element[i])
        y.append(element[n])
    return [x,y]

def plot_pca_scatter(name,X_pca, y_digits, q, n):
    colors = ['yellow', 'cyan']
    for i in xrange(len(colors)):
        px = X_pca[:, q][y_digits == i]
        py = X_pca[:, n][y_digits == i]
        plt.scatter(px, py, c=colors[i])
    plt.xlabel(str(q) + ' Principal Component')
    plt.ylabel(str(n) + ' Principal Component')
    plt.title(name)
    plt.show() 

def gmm_results(title, A, B):
    X_train = A[0]
    y_train = A[1]
    X_test = B[0]
    y_test = B[1]
    n_classes = len(np.unique(y_train))
    # Try GMMs using different types of covariances.
    classifiers = dict((covar_type, GMM(n_components=n_classes,
                        covariance_type=covar_type, init_params='wc', n_iter=20))
                       for covar_type in ['spherical', 'diag', 'tied', 'full'])

    n_classifiers = len(classifiers)

    plt.figure(figsize=(3 * n_classifiers / 2, 6))
    plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                        left=.01, right=.99)


    for index, (name, classifier) in enumerate(classifiers.items()):
        # Since we have class labels for the training data, we can
        # initialize the GMM parameters in a supervised manner.
        classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                      for i in xrange(n_classes)])

        start = time.time()
        # Train the other parameters using the EM algorithm.
        classifier.fit(X_train)
        end = time.time()
        print 'Fit Time: ' + str(end - start)

        h = plt.subplot(2, n_classifiers / 2, index + 1)
        make_ellipses(classifier, h)

        for n, color in enumerate('rg'):
            data = X_train[y_train == n]
            plt.scatter(data[:, 0], data[:, 1], 0.8, color=color)
        # Plot the test data with crosses
        for n, color in enumerate('rgb'):
            data = X_test[y_test == n]
            plt.plot(data[:, 0], data[:, 1], 'x', color=color)

        y_train_pred = classifier.predict(X_train)
        ld.save_data('datasets/' + name.replace(' ', '_') + '_train.csv', [y_train_pred,y_train])
        train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
        print 'Train Accuracy' + str(train_accuracy)
        plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
                 transform=h.transAxes)

        y_test_pred = classifier.predict(X_test)
        ld.save_data('datasets/' + name.replace(' ', '_') + '_test.csv', [y_test_pred, y_test])
        test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
        print 'Test Accuracy' + str(test_accuracy)
        plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
                 transform=h.transAxes)

        plt.xticks(())
        plt.yticks(())
        plt.title(name)

    plt.legend(loc='lower right', prop=dict(size=12))

    plt.savefig('figures/' + title.replace(' ', '_') + '_Training_results.png')
    #plt.show()
    
ld = Loader()
#np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]  
 

[X, y] = ld.load_data('datasets/Curious_George_train_features_100_percent.csv')
[X_test, y_test] = ld.load_data('datasets/Curious_George_test_features.csv')

# You are to implement (or find the code for) six algorithms. The first two are clustering algorithms:
    # Expectation Maximization

# The last four algorithms are dimensionality reduction algorithms:
    # PCA
    # ICA
    # Randomized Projections
    # Any other feature selection algorithm you desire

# You are to run a number of experiments. Come up with at least two datasets. If you'd like (and it makes a lot of sense in this case) you can use the ones you used in the first assignment.

    # Run the clustering algorithms on the data sets and describe what you see.
    # Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
    # Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it.
    # Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 (if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've already done this) and rerun your neural network learner on the newly projected data.
    # Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction algorithms (you've probably already done this), treating the clusters as if they were new features. In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. Again, rerun your neural network learner on the newly projected data.

start = time.time()
print "Standardize Data"
stdsc = StandardScaler()
X_scaled = stdsc.fit_transform(X)
end = time.time()
X_test_scaled = stdsc.transform(X_test)
print end - start
gmm_results('GMM Curious George Standardized Data, No Feature Selection',[X_scaled,y], [X_test_scaled, y_test])


print "Apply the dimensionality reduction algorithms to the two datasets and describe what you see."
stdsc = StandardScaler()
pca = decomposition.PCA(n_components=2)
X_pca = stdsc.fit_transform(X)
X_test_pca = stdsc.transform(X_test)
start = time.time()
X_pca = pca.fit_transform(X_pca)
end = time.time()
X_test_pca = pca.transform(X_test_pca)

print end - start
gmm_results('GMM Curious George PCA Feature Selection ', [X_pca,y], [X_test_pca, y_test])   



print "Fast ICA Data "
stdsc = StandardScaler()
ica = decomposition.FastICA(n_components=2)

X_ica = stdsc.fit_transform(X)
X_test_ica = stdsc.transform(X_test)
start = time.time()
X_ica = ica.fit_transform(X_ica)
end = time.time()
print end - start

X_test_ica = ica.transform(X_test_ica)
gmm_results('GMM Curious George ICA Feature Selection', [X_ica,y], [X_test_ica, y_test])



print "Random Projection Data components"
stdsc = StandardScaler()
rp = random_projection.GaussianRandomProjection(n_components=2)
X_rp = stdsc.fit_transform(X)
X_test_rp = stdsc.transform(X_test)
start = time.time()
X_rp = rp.fit_transform(X_rp)
end = time.time()
print end - start
X_test_rp = rp.transform(X_test_rp)
gmm_results('GMM Curious George RP Feature Selection', [X_rp,y], [X_test_rp, y_test])



print "Kernel PCA Data components"
stdsc = StandardScaler()
pca = decomposition.KernelPCA(n_components=2)
X_pca = stdsc.fit_transform(X)
X_test_pca = stdsc.transform(X_test)
start = time.time()
X_pca = pca.fit_transform(X_pca)
end = time.time()
print end - start
X_test_pca = pca.transform(X_test_pca)
gmm_results('GMM Curious George Kernel PCA Feature Selection', [X_pca,y], [X_test_pca, y_test])



print "Random PCA Data components"
stdsc = StandardScaler()
pca = decomposition.RandomizedPCA(n_components=2)
X_pca = stdsc.fit_transform(X)
X_test_pca = stdsc.transform(X_test)
start = time.time()
X_pca = pca.fit_transform(X_pca)
end = time.time()
print end - start
X_test_pca = pca.transform(X_test_pca)
gmm_results('GMM Curious George Random PCA Feature Selection', [X_pca,y], [X_test_pca, y_test])


