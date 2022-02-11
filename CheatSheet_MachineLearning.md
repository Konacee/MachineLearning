# Cheat  Sheet Machine Learning
## DataFrames
- preprocessing
   - df.drop([' '], axis=1)
   - pd.get_dummies(df) #one-hot encodes the data, so it's all 1's and 0's
   - df.fillna(0.0) #fill the blanks
- read
   - pd.read_csv(' ')
   - pd. smt with table
- access
   - df[str/int]
   - df.loc['str']
   - df.iloc[[int/intrange], [int/intrange]]
## Regression


 
 ## Classification
 
 
 ## Decisiontree
 DecisionTreeClassifier()
   - max_depth
   - min_leaf_size
   - min_leaf_split
 
 
 
###### Code Example

```
from sklearn.tree import DecisionTreeClassifier


model = DecisionTreeClassifier(max_depth=7)
model.fit(X,y)


y_pred = model.predict(X)
acc = accuracy_score(y,y_pred)
```
## Naive Bayes
CountVectorizer()
   - .fit_transform(X_train)
   - .transform(x_test)
   
   -> use these for generating the training and testing data before fitting the model

MultinomialNB()
###### Code Example
```
# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

# Instantiate our model
naive_bayes = MultinomialNB()

# Fit our model to the training data
naive_bayes.fit(training_data, y_train)
```

## Gridsearch
GridSearchCV
###### Code Example
```
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold


model_new = DecisionTreeClassifier()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)

space = dict()
space['max_depth'] = [7, 8, 9, 10]
space['min_samples_leaf'] = [4, 5, 6, 7, 8]
space['min_samples_split'] = [12, 13, 14, 15, 16, 17]

search = GridSearchCV(model_new, space, scoring='accuracy', n_jobs=-1, cv=cv)

result_new = search.fit(X_train, y_train)

print('Best Score: %s' % result_new.best_score_)
print('Best Hyperparameters: %s' % result_new.best_params_)
```

## Support Vector Machine (SVC)
SVC()
   - C: The C parameter. higher C = less errors & small margin
   - kernel: The kernel. The most common ones are 'linear', 'poly', and 'rbf'.
   - degree: If the kernel is polynomial, this is the maximum degree of the monomials in the kernel.
   - gamma : If the kernel is rbf, this is the gamma parameter.

## Ensembles
AdaBoostClassifier()
BaggingClassifier()
   - Base_Estimator(model(' ')): The model utilized for the weak learners (If None, then the base estimator is DecisionTreeClassifier initialized with max_depth=1)
   - n_estimators: The maximum number of weak learners used

RandomForestClassifier()
   - n_estimators
   - criterion: gini (gini-impurity) or entropy (imformation gain)
   - max_depth, min_samples, etc.


## Sklearn general

##### train_test_split
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size=0.2, random_state=42)
```
##### accuracy score

```
from sklearn.metrics import accuracy_score
sklearn.metrics.accuracy_score(y_true, y_pred)
```
