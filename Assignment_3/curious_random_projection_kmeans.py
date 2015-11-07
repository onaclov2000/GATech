import time
from loader import Loader
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn import datasets
from sklearn import random_projection
from sklearn import mixture
import itertools
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
    
def k_means_results(name, A, B, x_label, y_label, colormap):
    X = A[0]
    y = A[1]
    X_test = B[0]
    y_test = B[1]
    h = .02
    n_clusters = 2
    k_means = KMeans(n_clusters=n_clusters)
    start = time.time()
    fit_results = k_means.fit(X)
    end = time.time()
    print 'Fit Time: ' + str(end - start)
    Y_kmeans = k_means.predict(X)
    ld.save_data('datasets/' + name.replace(' ', '_') + '_train.csv', [Y_kmeans,y])
    # print Y_kmeans
    plt.figure()
    colors = ['yellow', 'cyan']
    if colormap:
        cmap_light = ListedColormap(['#FF3EFA', '#AAFFAA'])
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    for i in xrange(len(colors)):
        px = X[:, 0][Y_kmeans == i]
        py = X[:, 1][Y_kmeans == i]
        plt.scatter(px, py, c=colors[i])
    plt.scatter(fit_results.cluster_centers_[0, 0:1],fit_results.cluster_centers_[0, 1:2] , s=100, linewidths=4, c='orange', marker='x')
    plt.scatter(fit_results.cluster_centers_[1, 0:1],fit_results.cluster_centers_[1, 1:2] , s=100, linewidths=4, c='orange', marker='x')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(name + ' Train Results')
#    plt.show()
    plt.savefig('figures/' + name.replace(' ', '_') + '_Training_results.png')
    plt.clf()

    y_pred = Y_kmeans
    y_true = y

    print 'Accuracy Score'
    print metrics.accuracy_score(y_true, y_pred)
    print 'Classification Report'
    print metrics.classification_report(y_true, y_pred)
    print 'Confusion Matrix'
    print metrics.confusion_matrix(y_true, y_pred)
    print 'Completeness Score'
    print metrics.completeness_score(y_true,y_pred)
    print 'Homogeneity Score'
    print metrics.homogeneity_score(y_true,y_pred)
    print 'Homogeneity Completeness V Measured'
    print metrics.homogeneity_completeness_v_measure(y_true,y_pred)
    print 'Mutual Information Score'
    print metrics.mutual_info_score(y_true,y_pred)
    print 'Normalized Mutual Info Score'
    print metrics.normalized_mutual_info_score(y_true,y_pred)
    print 'Silhouette Score'
    print metrics.silhouette_score(X,fit_results.labels_)
    print 'Silhouette Samples'
    print metrics.silhouette_samples(X,fit_results.labels_)
    print 'V Measure Score'
    print metrics.v_measure_score(y_true,y_pred)

    print_confusion_matrix('Train', Y_kmeans, y)
    
    Y_kmeans = k_means.predict(X_test)
    ld.save_data('datasets/' + name.replace(' ', '_') + '_test.csv', [Y_kmeans,y_test])
    colors = ['yellow', 'cyan']
    if colormap:
        cmap_light = ListedColormap(['#FF3EFA', '#AAFFAA'])
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    for i in xrange(len(colors)):
        px = X_test[:, 0][Y_kmeans == i]
        py = X_test[:, 1][Y_kmeans == i]
        plt.scatter(px, py, c=colors[i])
    plt.scatter(fit_results.cluster_centers_[0, 0:1],fit_results.cluster_centers_[0, 1:2] , s=100, linewidths=4, c='orange', marker='x')
    plt.scatter(fit_results.cluster_centers_[1, 0:1],fit_results.cluster_centers_[1, 1:2] , s=100, linewidths=4, c='orange', marker='x')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(name + ' Test Results')
#    plt.show()
    plt.savefig('figures/' + name.replace(' ', '_') + '_Test_results.png')
    print_confusion_matrix('Test', Y_kmeans, y_test)    

def split_data(a,i,n):
    x = []
    y = []
    for element in a:
        x.append(element[i])
        y.append(element[n])
    return [x,y]

def plot_scatter(name,classifier,X_pca, y_digits, x_label, y_label):
    colors = ['yellow', 'cyan']
    plt.figure('')
    for i in xrange(len(colors)):
        px = X_pca[:, 0][y_digits == i]
        py = X_pca[:, 1][y_digits == i]
        plt.scatter(px, py, c=colors[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(name)
    plt.savefig('figures/' + name.replace(' ', '_') + classifier + '.png')

    
    
ld = Loader()
#np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]  
 

[X, y] = ld.load_data('datasets/Curious_George_train_features_100_percent.csv')
[X_test, y_test] = ld.load_data('datasets/Curious_George_test_features.csv')


# You are to implement (or find the code for) six algorithms. The first two are clustering algorithms:

    # k-means clustering
    # Expectation Maximization

# You can choose your own measures of distance/similarity. Naturally, you'll have to justify your choices, but you're practiced at that sort of thing by now.

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

print "Run the clustering algorithms on the data sets and describe what you see."
k_means_results('KMeans Curious George No Feature Selection', [X,y], [X_test, y_test], '1st Feature', '2nd Feature', colormap = False)

for i in range(0,100):
	print "Random Projection Data components"
	stdsc = StandardScaler()
	rp = random_projection.GaussianRandomProjection(n_components=2)
	X_rp = stdsc.fit_transform(X)
	X_test_rp = stdsc.transform(X_test)
	start = time.time()
	X_rp = rp.fit_transform(X_rp)
	end = time.time()
	print "Fit Time: " + str(end - start)
	X_test_rp = rp.transform(X_test_rp)
	k_means_results('KMeans Curious George RP Feature Selection ' + str(i), [X_rp,y], [X_test_rp, y_test],  '1st RP Component', '2nd RP Component', colormap = True)
	plot_scatter('KMeans Curious George Feature Selection ' + str(i), 'RP', X_rp, y, '1st RP Component', '2nd RP Component')
	ld.save_data('datasets/Curious_George_train_features_100_percent_random_projection_'  + str(i) + '.csv', [X_rp, y])
	ld.save_data('datasets/Curious_George_test_features_random_projection_'  + str(i) + '.csv', [X_test_rp, y_test])



