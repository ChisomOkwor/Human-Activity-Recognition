#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_decision_regions


from sklearn import tree

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import datetime

from sklearn import linear_model
from sklearn import metrics
from sklearn import linear_model
from sklearn import metrics

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# ## Human Activity Recognition Using Smartphones Data Set
# ### Data gotten from https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
# 
# ### Human Activity Recognition database built from the recordings of 30 subjects performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors.
# 
# ### Classifies 6 different exercises STANDING  SITTING, WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS,  LAYING
# 

# In[2]:


# Load datasets. Note: Training and Testing is Separate 
X_train = pd.read_table(
             'train/X_train.txt', sep='\s+', header=None)
y_train = pd.read_table(
             'train/y_train.txt', sep=' ', header=None)

X_test = pd.read_table(
             'test/X_test.txt', sep='\s+', header=None)
y_test = pd.read_table(
             'test/y_test.txt', sep=' ', header=None)

activities_label = pd.read_table(
            'activity_labels.txt',   
            sep=' ',                 
            header=None,              
            names=('ID','Activity'))  
features = pd.read_table(
                'features.txt', sep=' ',
                header=None, names=('ID','Sensor'))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
X_train


# In[3]:


X_test


# In[4]:


# Merge Training and Testing data for both dependent and Independent
allX = pd.concat([X_train, X_test], ignore_index = True)
allY = y_train.append(y_test, ignore_index=True)

# Name the column names with data from different DF
sensorNames = features['Sensor']  # Get the names from features DF
allX.columns = sensorNames  # Change column names

print(allX.shape, allY.shape)


# In[5]:


# Add the dependent varible column: Activity
for i in activities_label['ID']:
  activity = activities_label[activities_label['ID'] == i]['Activity'] # get activity cell given ID
  allY = allY.replace({i: activity.iloc[0]}) # replace this ID with activity string
 
allY.columns = ['Activity'] # change column name from ID to Activity


# #### Merge and save the pre-split data

# In[6]:


tidy_data = pd.concat([allX, allY], axis=1)
tidy_data.to_csv("tidyHARdata.csv")
print(tidy_data.shape)
tidy_data


# ## Pre-process the Data

# In[7]:


# The columns are all floats except the activity column (Dependent Column)
tidy_data.dtypes.value_counts()


# In[8]:


# There are no missing entries in all features
tidy_data.isnull().values.any()


# In[9]:


# Dependent variable Y is categorical
tidy_data.Activity.value_counts()


# #### Encode the dependent varible 
# #### LabelEncoder needs to be used to convert the activity(dependent) labels to integers

# In[10]:


le = LabelEncoder()
tidy_data['Activity'] = le.fit_transform(tidy_data.Activity) 


# In[11]:


le.inverse_transform([0,1,2,3,4,5]) # To know which one corresponds to each number
le.transform(['LAYING'])


# #### SPLIT THE DATASET 

# In[12]:


tidy_data.head()


# In[13]:


X, y = tidy_data.iloc[:, :561].values, tidy_data.iloc[:, 561].values
print(X.shape, y.shape)
print(X)
print(y)


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=None, stratify=y)


# ### Logistic Regression

# In[15]:


lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)
print('Test Accuracy: %.3f' % lr.score(X_test, y_test))


# ### Support Vector Machine

# In[16]:


sv = svm.SVC().fit(X_train, y_train)
print('Test Accuracy: %.3f' % sv.score(X_test, y_test))


# ### Multiple Layer Peceptron (MLP)

# In[17]:


mlp = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
print('Test Accuracy: %.3f' % mlp.score(X_test, y_test))


# ### Decision Tree

# In[18]:


dt = tree.DecisionTreeClassifier().fit(X_train, y_train)
print('Test Accuracy: %.3f' % dt.score(X_test, y_test))


# ### Ensemble Learning with Random Forest

# In[19]:


rf = RandomForestClassifier().fit(X_train, y_train)
print('Test Accuracy: %.3f' % rf.score(X_test, y_test))


# ### Utils 

# In[20]:


from datetime import datetime
labels=['LAYING', 'SITTING','STANDING','WALKING','WALKING_DOWNSTAIRS','WALKING_UPSTAIRS']

def perform_model(model, X_train, y_train, X_test, y_test, class_labels, cm_normalize=True,                  print_cm=True, cm_cmap=plt.cm.Greens):
    
    
    # to store results at various phases
    results = dict()
    
    # time at which model starts training 
    train_start_time = datetime.now()
    print('training the model..')
    model.fit(X_train, y_train)
    print('Done....!\n')
    train_end_time = datetime.now()
    results['training_time'] =  train_end_time - train_start_time
    print('==> training time:- {}\n'.format(results['training_time']))
    
    
    # predict test data
    print('Predicting test data')
    test_start_time = datetime.now()
    y_pred = model.predict(X_test)
    test_end_time = datetime.now()
    print('Done....!\n')
    results['testing_time'] = test_end_time - test_start_time
    print('==> testing time:- {}\n'.format(results['testing_time']))
    results['predicted'] = y_pred


    # calculate overall accuracty of the model
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    # store accuracy in results
    results['accuracy'] = accuracy
    print('==> Accuracy:- {}\n'.format(accuracy))
   
    # confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm: 
        print('\n ********Confusion Matrix********')
        print('\n {}'.format(cm))
        
    # plot confusin matrix
    plt.figure(figsize=(6,6))
    plt.grid(b=False)
    plot_confusion_matrix(cm, classes=class_labels, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
    plt.show()

    
    print('\n ********Recall, Precision and F1 Score ********')
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
    
    # get classification report
    print('****************| Classifiction Report |****************')
    classification_report = metrics.classification_report(y_test, y_pred)
    
     # store report in results
    results['classification_report'] = classification_report
    print(classification_report)
    
    # add the trained  model to the results
    results['model'] = model
    
    return results
    
   


# In[21]:


def print_grid_search_attributes(model):
    # Estimator that gave highest score among all the estimators formed in GridSearch
    print('\n\n==> Best Estimator:')
    print('\t{}\n'.format(model.best_estimator_))


    # parameters that gave best results while performing grid search
    print('\n==> Best parameters:')
    print('\tParameters of best estimator : {}'.format(model.best_params_))


    #  number of cross validation splits
    print('\n==> No. of CrossValidation sets:')
    print('\tTotal numbre of cross validation sets: {}'.format(model.n_splits_))


    # Average cross validated score of the best estimator, from the Grid Search 
    print('\n==> Best Score:')
    print('\tAverage Cross Validate scores of best estimator : {}'.format(model.best_score_))


# In[22]:


plt.rcParams["font.family"] = 'DejaVu Sans'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ### Logistic Regression with Grid Search

# In[23]:


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

# start Grid search
parameters = {'C':[0.01, 0.1, 1, 10, 20, 30], 'penalty':['l2','l1']}
log_reg = linear_model.LogisticRegression()
log_reg_grid = GridSearchCV(log_reg, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
log_reg_grid_results =  perform_model(log_reg_grid, X_train, y_train, X_test, y_test, class_labels=labels)

# observe the attributes of the model 
print_grid_search_attributes(log_reg_grid_results['model'])


# ### Decision Tree with Grid Search

# In[ ]:


parameters = {'max_depth':np.arange(3,10,2)}
dt = DecisionTreeClassifier()
dt_grid = GridSearchCV(dt,param_grid=parameters, n_jobs=-1)
dt_grid_results = perform_model(dt_grid, X_train, y_train, X_test, y_test, class_labels=labels)

# observe the attributes of the model 
print_grid_search_attributes(dt_grid_results['model'])


# ###  Support Vector Machine with Grid Search

# In[ ]:


parameters = {'C':[2,8,16],              'gamma': [ 0.0078125, 0.125, 2]}
rbf_svm = svm.SVC(kernel='rbf')
rbf_svm_grid = GridSearchCV(rbf_svm, param_grid=parameters, n_jobs=-1)
rbf_svm_grid_results = perform_model(rbf_svm_grid, X_train, y_train, X_test, y_test, class_labels=labels)

# observe the attributes of the model 
print_grid_search_attributes(rbf_svm_grid_results['model'])


# ### MLP with Grid Search 

# In[ ]:


parameters = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

mlp_grid =  MLPClassifier()
mlp_grid = GridSearchCV(mlp_grid, param_grid=parameters, n_jobs=-1, cv=5)
mlp_grid_results = perform_model(mlp_grid, X_train, y_train, X_test, y_test, class_labels=labels)

# observe the attributes of the model 
print_grid_search_attributes(mlp_grid_results['model'])


# ### Ensemble Learning, Random Forest with Grid Search

# In[ ]:


rf = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True)
param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_grid = GridSearchCV(rf, param_grid=param_grid, cv=5)
rf_grid_results = perform_model(rf_grid, X_train, y_train, X_test, y_test, class_labels=labels)

# Observe the attributes of the model 
print_grid_search_attributes(rf_grid_results['model'])


# In[ ]:


from matplotlib.colors import ListedColormap

# def plot_decision_regions(X, y, classifier, resolution=0.02):

#     # setup marker generator and color map
#     markers = ('s', 'x', 'o', '^', 'v')
#     colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#     cmap = ListedColormap(colors[:len(np.unique(y))])

#     # plot the decision surface
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#                            np.arange(x2_min, x2_max, resolution))
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)
#     plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())

#     # plot examples by class
#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[y == cl, 0], 
#                     y=X[y == cl, 1],
#                     alpha=0.6, 
#                     color=cmap(idx),
#                     edgecolor='black',
#                     marker=markers[idx], 
#                     label=cl)


# In[ ]:


pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
principalDf = pd.DataFrame(data = X_train_pca, columns = ['P1', 'P2'])
finalDf = pd.concat([principalDf, pd.DataFrame(y, columns=['target'])], axis = 1)

X_test_pca = pca.transform(X_test)

lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
lr = lr.fit(X_train_pca, y_train)
print('Test Accuracy: %.3f' % lr.score(X_test_pca, y_test))


# In[ ]:


fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('P1', fontsize = 15)
ax.set_ylabel('P2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
markers = ['x', 's']

for target, color, m in zip(targets,colors, markers):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'P1'], finalDf.loc[indicesToKeep, 'P2'], c = color, s = 30, marker=m)
ax.legend(targets)
ax.grid()


# In[ ]:


# plot_decision_regions(X_train_pca, y_train, classifier=lr)
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train_pca, y_train, clf=lr, legend=2)


plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_04.png', dpi=300)
plt.show()


# In[ ]:


# plot_decision_regions(X_test_pca, y_test, classifier=lr)
plot_decision_regions(X_test_pca, y_train, clf=lr, legend=2)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_05.png', dpi=300)
plt.show()


# In[ ]:


pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train)
pca.explained_variance_ratio_


# In[ ]:





# ### Draw ROC AOC CURVE

# In[ ]:


# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import roc_curve, auc

# y_score = lr.decision_function(X_test)
# y = label_binarize(y, classes=[0, 1, 2])

# n_classes = y.shape[1]


# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# ### Draw Learning Curve for best model

# In[ ]:


# Learning Curve using the Grid Search Estimator
train_sizes, train_scores, test_scores = learning_curve(estimator=lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=8,)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='red', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='red')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()


# In[ ]:





# In[ ]:




