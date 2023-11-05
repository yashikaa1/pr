from sklearn import datasets 
iris=datasets.load_iris()
print (iris.target_names)
print (iris.feature_names)
print (iris.data[0:5])
print (iris.target)
import pandas as pd 
data=pd.DataFrame({ 
    'sepal length': iris.data[:,0], 
    'sepal width': iris.data[:,1], 
    'petal length': iris.data[:,2], 
    'petal width':iris.data[:,3], 
    'species': iris.target}) 
data.head()
from sklearn.model_selection import train_test_split
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']] 
y=data['species']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3) 
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier (n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn import metrics
print("Accuracy:", metrics.accuracy_score (y_test, y_pred)) 
ans = clf.predict([[3, 5, 4, 2]])
if ans [0]==0:
    print('setosa') 
elif ans [0]==1:
    print('versicolor')
else:
    print('virginica')