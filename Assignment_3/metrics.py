import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from loader import Loader
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
ld = Loader()
[X, y] = ld.load_data('datasets/Curious_George_train_features_100_percent.csv')
#[X_test, y_test] = ld.load_data('datasets/Curious_George_test_features.csv')
def makebin(a):
	return 1.0 == a
y_true = map(makebin, y)

k_means = KMeans(n_clusters=2)
result = k_means.fit(X)
y_pred = k_means.predict(X) 

accuracy_score = []
#http://scikit-learn.org/stable/modules/classes.html
print 'Accuracy Score'
accuracy_score.append(metrics.accuracy_score(y_true, y_pred))
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
print metrics.silhouette_score(X,result.labels_)
print 'Silhouette Samples'
print metrics.silhouette_samples(X,result.labels_)
print 'V Measure Score'
print metrics.v_measure_score(y_true,y_pred)


stdsc = StandardScaler()
X_scaled = stdsc.fit_transform(X)
k_means = KMeans(n_clusters=2)
result = k_means.fit(X_scaled)
y_pred = k_means.predict(X_scaled) 


#http://scikit-learn.org/stable/modules/classes.html
print 'Accuracy Score'
accuracy_score.append(metrics.accuracy_score(y_true, y_pred)
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
print metrics.silhouette_score(X,result.labels_)
print 'Silhouette Samples'
print metrics.silhouette_samples(X,result.labels_)
print 'V Measure Score'
print metrics.v_measure_score(y_true,y_pred)

stdsc = StandardScaler()
X_scaled = stdsc.fit_transform(X)
pca = decomposition.PCA()
X_pca = stdsc.fit_transform(X)
pca.fit(X_pca)
k_means = KMeans(n_clusters=2)
result = k_means.fit(X_pca)
y_pred = k_means.predict(X_pca) 
#http://scikit-learn.org/stable/modules/classes.html
print 'Accuracy Score'
accuracy_score.append(metrics.accuracy_score(y_true, y_pred))
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
print metrics.silhouette_score(X,result.labels_)
print 'Silhouette Samples'
print metrics.silhouette_samples(X,result.labels_)
print 'V Measure Score'
print metrics.v_measure_score(y_true,y_pred)



