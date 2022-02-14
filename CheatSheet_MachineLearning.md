# Cheat  Sheet Machine Learning
## DataFrames
- preprocessing
   - df.drop([' '], axis=1)
   - pd.get_dummies(df) #one-hot encodes the data, so it's all 1's and 0's
   - df.fillna(0.0) #fill the blanks
   - df['label'] = df.label.map({'ham':0, 'spam':1}) #encode names
   - CountVectorizer() #convert complex data to numbers, see Naive Bayes
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
recall, precission, etc.

## Statistics
http://mlwiki.org/index.php/Precision_and_Recall

#### ROC Score and AUC
##### Code Example
```
# Function for calculating auc and roc

def build_roc_auc(model, X_train, X_test, y_train, y_test):
    '''
    INPUT:
    model - an sklearn instantiated model
    X_train - the training data
    y_train - the training response values (must be categorical)
    X_test - the test data
    y_test - the test response values (must be categorical)
    OUTPUT:
    auc - returns auc as a float
    prints the roc curve
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from scipy import interp
    
    y_preds = model.fit(X_train, y_train).predict_proba(X_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(y_test)):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_preds[:, 1])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_preds[:, 1].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.show()
    
    return roc_auc_score(y_test, np.round(y_preds[:, 1]))
    
    
# Finding roc and auc for the random forest model    
build_roc_auc(rf_mod, training_data, testing_data, y_train, y_test) 
```

