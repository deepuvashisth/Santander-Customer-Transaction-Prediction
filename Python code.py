import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')

data.isnull().sum()

data.describe()

del data['ID_code']

data['target'] = data['target'].astype('category')


#Feature Selection
x = data.values[:,1:202]
x = pd.DataFrame(x)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=50, step=10, verbose=5)
rfe_selector.fit(data.values[:,1:202], data['target'])
rfe_support = rfe_selector.get_support()
rfe_feature = x.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')

df = data.iloc[: , rfe_feature]
test = test.iloc[: , rfe_feature]

#Outlier Analysis
from scipy import stats

def drop_numerical_outliers(df, z_thresh=3):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh) \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)

drop_numerical_outliers(df)


#Splitting in train and test
x = df.values[:,1:50]
y = df.values[:,0]

x = pd.DataFrame(x)
y = pd.DataFrame(y)

y = y.astype(int)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


#Decision Tree
from sklearn import tree
dt_model = tree.DecisionTreeClassifier(max_depth=1).fit(x_train, y_train)

dt_predicts = dt_model.predict(x_test)


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 

confusion_matrix(y_test, dt_predicts) 
accuracy_score(y_test, dt_predicts) 
#Accuracy = 0.899
#Precision = 1.000
#Recall = 0.898


#Random Forest
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 100).fit(x_train, y_train)

rf_predict = RF_model.predict(x_test)

confusion_matrix(y_test, rf_predict) 
accuracy_score(y_test, rf_predict) 
#Accuracy = 0.899
#Precision = 0.999
#Recall = 0.898


#KNN Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 3).fit(x_train, y_train)

knn_predicts = knn_model.predict(x_test)

confusion_matrix(y_test, knn_predicts) 
accuracy_score(y_test, knn_predicts) 
#Accuracy = 0.887
#Precision = 0.985
#Recall = 0.899


#Logistic Regression
xx = x_train
import statsmodels.api as sm
logit_model=sm.Logit(y_train,xx).fit()
logit_model.summary2()

xx = xx.drop(axis = 1, columns = [4, 8, 10, 22, 33, 40])

from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression().fit(xx, y_train)

lg_predicts = lg_model.predict(x_test.drop(axis = 1, columns = [4, 8, 10, 22, 33, 40]))

confusion_matrix(y_test, lg_predicts) 
accuracy_score(y_test, lg_predicts) 
#Accuracy = 0.898
#Precision = 0.999
#Recall = 0.898



target = dt_model.predict(test.iloc[:,1:50])

test['target'] = target

test.to_csv('target.csv')