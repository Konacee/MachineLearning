# Cheat  Sheet Machine Learning
## Regression


 
 ## Classification
 
 
 ## Decisiontree
 - DecisionTreeClassifier()
    - max_depth
    - min_leaf_size
    - min_leaf_split
 - accuracy_score(y,y_pred)
 
###### Code Example

```
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

model = DecisionTreeClassifier(max_depth=7)
model.fit(X,y)


y_pred = model.predict(X)
acc = accuracy_score(y,y_pred)
```

## Sklearn

##### train_test_split
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size=0.2, random_state=42)
```
