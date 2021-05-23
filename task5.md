# 异常检测
## 高维数据异常检测

### 使用Pyod库生成toy example并调用feature bagging
from pyod.models.feature_bagging import FeatureBagging
from pyod.utils.data import generate_data,evaluate_print
from pyod.utils.example import visualize
contamination = 0.1  # percentage of outliers
n_train = 200  # number of training points
n_test = 100  # number of testing points
X_train, y_train, X_test, y_test = generate_data(n_train=n_train, n_test=n_test, contamination=contamination)
clf_name = 'FeatureBagging'
clf = FeatureBagging()
clf.fit(X_train)
# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores
# get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores
# evaluate and print the results
print("\nFeatureBagging On Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nFeatureBagging On Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)

#FeatureBagging On Training Data:
#FeatureBagging ROC:0.9681, precision @ rank n:0.9

#FeatureBagging On Test Data:
#FeatureBagging ROC:0.9989, precision @ rank n:0.9
![image](https://user-images.githubusercontent.com/33819026/119264150-80ba6880-bc14-11eb-8574-9337bcb2a6d4.png)

### 使用Pyod库生成toy example并调用Isolation Forests

from pyod.models.iforest import IForest
from pyod.utils.data import generate_data,evaluate_print
from pyod.utils.example import visualize
contamination = 0.1  # percentage of outliers
n_train = 200  # number of training points
n_test = 100  # number of testing points
X_train, y_train, X_test, y_test = generate_data(n_train=n_train, n_test=n_test, contamination=contamination)

clf_name = 'IForest'
clf = IForest()
clf.fit(X_train)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores
 
# get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores

# evaluate and print the results
print("\nIForest On Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nIForest On Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)

#IForest On Training Data:
#IForest ROC:0.9897, precision @ rank n:0.95
#IForest On Test Data:
#IForest ROC:1.0, precision @ rank n:1.0
#可视化所有示例
visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
          y_test_pred, show_figure=True, save_figure=False)
![image](https://user-images.githubusercontent.com/33819026/119264169-9891ec80-bc14-11eb-81e5-e52f1cc3ed00.png)
