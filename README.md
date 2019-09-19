# awesome-titanic-analysis-prediction
Analysis and classification of the Titanic dataset

This repo demonstrates some data visualization, preparation and classification methods for the Titanic dataset. The Titanic Dataset is one of the most famous datasets available, thanks to its wealth of features to analyze and easily definable target, and it can be found [here](https://www.kaggle.com/c/titanic). The goal of this repo is to assist people in learning how to prepare data and implement classifiers. To that end, there is also a IPython notebook included with this repo.

Here is a brief breakdown of the sections of code.

The first section handles importing and preprocessing of the data.

```Python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
warnings.filterwarnings("ignore")

# load in the train and test data
training_data = pd.read_csv("train.csv")
testing_data = pd.read_csv("test.csv")

print(training_data.head(10))
print(testing_data.head(10))

# let's see what features are availalbe to us
print(training_data.keys())
print(testing_data.keys())

# have to find out what our numerical features are
train_datatypes = training_data.dtypes
print(train_datatypes)
print(training_data.describe())

def get_nulls(training, testing):
    print("Training Data:")
    print(pd.isnull(training).sum())
    print("Testing Data:")
    print(pd.isnull(testing).sum())

get_nulls(training_data, testing_data)
          
# let's drop the cabin column, because it has a lot of missing values
# ticket numbers contain far too many categories as well, so let's drop that too

training_data.drop(labels = ['Cabin', 'Ticket'], axis = 1, inplace=True)
testing_data.drop(labels = ['Cabin', 'Ticket'], axis = 1, inplace=True)
          
# the data is slightly right skewed, or the young ages have slightly more prominence
# taking the mean/average value would be affected by the skew
# so we shouldn't use the mean to impute, rather we should use the median value

training_data["Age"].fillna(training_data["Age"].median(), inplace = True)
testing_data["Age"].fillna(testing_data["Age"].median(), inplace = True)
training_data["Embarked"].fillna("S", inplace = True)
testing_data["Fare"].fillna(testing_data["Fare"].median(), inplace = True)

get_nulls(training_data, testing_data)

# now there should be no more missing values
print(training_data.head(10))
print(testing_data.head(10))
```

The next section covers several different methods of visualization and analysis, for the features of sex, class, and survival.

```Python
# now that the data has been prepped, let's do some visualization of the data
# let's examine some of the trends that exist between features

sns.barplot(x='Sex', y='Survived', data=training_data)
plt.title("Survival Rates Compared With Gender")
plt.show()

women_survived = training_data[training_data.Sex == "female"]["Survived"].sum()
men_survived = training_data[training_data.Sex == "male"]["Survived"].sum()

print("Total survivors:" + str(women_survived + men_survived))
print(men_survived)
print(women_survived)
print("Percentage of women survived:" + str(women_survived/(men_survived + women_survived) * 100))
print("Percentage of men survived:" + str(men_survived/(men_survived + women_survived) * 100))

# let's see how the class of passengers affected survival rate

sns.barplot(x="Pclass", y="Survived", data=training_data)
plt.ylabel("Rate of Survival")
plt.title("Survival According To Class")
plt.show()

# count the number of survivors in each class, and survivors total

class_1_survived = training_data[training_data.Pclass == 1]["Survived"].sum()
class_2_survived = training_data[training_data.Pclass == 2]["Survived"].sum()
class_3_survived = training_data[training_data.Pclass == 3]["Survived"].sum()
total_class_survived = class_1_survived + class_2_survived + class_3_survived

print("Prop. of Class 1 in survived:")
print(class_1_survived/total_class_survived)
print("Prop. of Class 2 in survived:")
print(class_2_survived/total_class_survived)
print("Prop. of Class 3 in survived:")
print(class_3_survived/total_class_survived)

# plot both class and gender survival rates

sns.barplot(x="Pclass", y="Survived", hue="Sex", data=training_data)
plt.ylabel("Survival rate")
plt.title("Survival rates based on Class and Gender")
```

This next section covers visualization of Age and Survival.

```Python
# age is a variable that has a lot of influence on survival rates
# start by selecting the people who have survived, along with their age

survived_ages = training_data[training_data.Survived == 1]["Age"]
perished_ages = training_data[training_data.Survived == 0]["Age"]

# time to create plots for these variables
# define the subplot

plt.subplot(1, 2, 1)
# plot the distribution
sns.distplot(survived_ages, kde=False)
# set the axis
plt.axis([0, 100, 0, 100])
plt.title("Survived")
plt.ylabel("Proportion")

plt.subplot(1, 2, 2)
# plot the distribution
sns.distplot(perished_ages, kde=False)
# set the axis
plt.axis([0, 100, 0, 100])
plt.title("Perished")

# adjust the subplots
plt.subplots_adjust(right=1.2)
plt.show()

# we can also do a scatterplot of sorts

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                figsize=(10, 6))

sns.stripplot(x="Survived", y="Age", data=training_data, jitter=True, ax=ax1)
sns.violinplot(x="Pclass", y="Survived", data=training_data, ax=ax2)
sns.pairplot(training_data)

plt.show()
```

This block handles feature scaling and transformation. It covers encoding features, scaling features, and dropping features.

```Python
# let's do some feature engineering
# sex and embarked status are non-numerical variables, we'll need to make them numerical values
# we can do this through one hot encoding

# let's check to see what some of the sex and embarked classes look like
print(training_data.sample(5))

encoder_1 = LabelEncoder()
# fit the encoder on the data
encoder_1.fit(training_data["Sex"])

# transform and replace the training data
training_sex_encoded = encoder_1.transform(training_data["Sex"])
training_data["Sex"] = training_sex_encoded
test_sex_encoded = encoder_1.transform(testing_data["Sex"])
testing_data["Sex"] = test_sex_encoded

encoder_2 = LabelEncoder()
encoder_2.fit(training_data["Embarked"])

training_embarked_encoded = encoder_2.transform(training_data["Embarked"])
training_data["Embarked"] = training_embarked_encoded
testing_embarked_encoded = encoder_2.transform(testing_data["Embarked"])
testing_data["Embarked"] = testing_embarked_encoded

# let's check to see if the transformation worked
print(training_data.sample(5))

# let's assume the name is going to be useless and drop it
training_data.drop("Name", axis = 1, inplace = True)
testing_data.drop("Name", axis = 1, inplace = True)

# remember that the scaler takes arrays, so any value we wish to reshape we need to turn into array to scale
ages_train = np.array(training_data["Age"]).reshape(-1, 1)
fares_train = np.array(training_data["Fare"]).reshape(-1, 1)
ages_test = np.array(testing_data["Age"]).reshape(-1, 1)
fares_test = np.array(testing_data["Fare"]).reshape(-1, 1)

scaler = StandardScaler()

training_data["Age"] = scaler.fit_transform(ages_train)
training_data["Fare"] = scaler.fit_transform(fares_train)
testing_data["Age"] = scaler.fit_transform(ages_test)
testing_data["Fare"] = scaler.fit_transform(fares_test)
```

Here's the selection of features and labels and division into training and testing sets.

```Python
# now to select our training and testing data
X_train = training_data.drop(labels=['PassengerId', 'Survived'], axis=1)
y_train = training_data['Survived']
X_test = testing_data.drop("PassengerId", axis=1)

print(X_train.head(5))

# do some training on the validation set
# make the train and test data from validation data

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=27)
```

Here's the selection and implementation of classifiers, making use of GridSearchCV.

```Python
svc_clf = SVC(probability=True)
svc_param = {"kernel": ["rbf", "linear"]}
grid_svc = GridSearchCV(svc_clf, svc_param)
grid_svc.fit(X_train, y_train)
svc_opt = grid_svc.best_estimator_

linsvc_clf = LinearSVC()
linsvc_param = {"fit_intercept": [True, False] ,"max_iter": [100, 250, 500, 1000]}
linsvc_grid = GridSearchCV(linsvc_clf, linsvc_param)
linsvc_grid.fit(X_train, y_train)
lin_opt = linsvc_grid.best_estimator_

rf_clf = RandomForestClassifier()
parameters_rf = {"n_estimators": [4, 6, 8, 10, 12, 14, 16], "criterion": ["gini", "entropy"], "max_features": ["auto", "sqrt", "log2"],
                 "max_depth": [2, 3, 5, 10], "min_samples_split": [2, 3, 5, 10]}
grid_rf = GridSearchCV(rf_clf, parameters_rf)
grid_rf.fit(X_train, y_train)
rf_opt = grid_rf.best_estimator_

logreg_clf = LogisticRegression()
parameters_logreg = {"penalty": ["l2"], "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                     "max_iter": [25, 50, 100, 200, 400]}
grid_logreg = GridSearchCV(logreg_clf, parameters_logreg)
grid_logreg.fit(X_train, y_train)
logreg_opt = grid_logreg.best_estimator_

knn_clf = KNeighborsClassifier()
parameters_knn = {"n_neighbors": [3, 5, 10, 15, 20], "weights": ["uniform", "distance"],
                  "leaf_size": [10, 20, 30, 45, 60]}
grid_knn = GridSearchCV(knn_clf, parameters_knn)
grid_knn.fit(X_train, y_train)
knn_opt = grid_knn.best_estimator_

bnb_clf = BernoulliNB()
bnb_params = {"alpha":[0.20, 0.50, 1.0]}
grd_bnb = GridSearchCV(bnb_clf, bnb_params)
grd_bnb.fit(X_train, y_train)
bnb_opt = grd_bnb.best_estimator_

gnb_clf = GaussianNB()
gnb_params = {}
grid_gnb = GridSearchCV(gnb_clf, gnb_params)
grid_gnb.fit(X_train, y_train)
gnb_opt = grid_gnb.best_estimator_

dt_clf = DecisionTreeClassifier()
parameters_dt = {"criterion": ["gini", "entropy"], "splitter": ["best", "random"], "max_features": ["auto", "log2", "sqrt"]}
grid_dt = GridSearchCV(dt_clf, parameters_dt)
grid_dt.fit(X_train, y_train)
dt_opt = grid_dt.best_estimator_

xg_clf = XGBClassifier()
parameters_xg = {"objective" : ["reg:linear"], "n_estimators" : [5, 10, 15, 20]}
grid_xg = GridSearchCV(xg_clf, parameters_xg)
grid_xg.fit(X_train, y_train)
xgb_opt = grid_xg.best_estimator_

mlp_clf = MLPClassifier()
parameters_mlp = {"solver": ["adam", "sgd"], "max_iter": [100, 200, 300],
                  "hidden_layer_sizes": [50, 100, 200], "activation": ["relu", "tanh"]}
grid_mlp = GridSearchCV(mlp_clf, parameters_mlp)
grid_mlp.fit(X_train, y_train)
mlp_opt = grid_mlp.best_estimator_
```

Here's the analysis of the performance of those classifiers.

```Python
classifiers = [svc_opt, lin_opt, rf_opt, logreg_opt, knn_opt, bnb_opt, gnb_opt, dt_opt, xgb_opt, mlp_opt]

metrics = ["Classifier", "Accuracy", "Log Loss", "F1"]
# create a dataframe to store the variables
log = pd.DataFrame(columns=metrics)

for clf in classifiers:
    clf.fit(X_train, y_train)
    classifier = clf.__class__.__name__

    print(classifier + " perfofrmance is:")
    
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print("Accuracy: {}".format(acc))
    
    preds = clf.predict(X_val)
    l_loss = log_loss(y_val, preds)
    print("Log Loss: {}".format(l_loss))
    
    preds = clf.predict(X_val)
    f1 = f1_score(y_val, preds)
    print("F1 Score: {}".format(f1))
    
    print("Next classifier...")
    print()
    
    logs = pd.DataFrame([[classifier, acc*100, l_loss, f1]], columns=metrics)
    log = log.append(logs)
    
sns.barplot(x='Accuracy', y='Classifier', data=log, palette="bright")
plt.xticks(rotation=90)
plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.barplot(x='Log Loss', y='Classifier', data=log, palette="bright")
plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()

sns.barplot(x='F1', y='Classifier', data=log, palette="bright")
plt.xlabel('F1 Score')
plt.title('Classifier F1 Score')
plt.show()
```

Finally, the best performing classifiers are chosen and a combined classifier is created.

```Python
voting_clf = VotingClassifier(estimators=[('SVC', svc_opt), ('GNB', gnb_opt), ('LogReg', logreg_opt), ('RF', rf_opt), ('MLP', mlp_opt)], voting='soft')
voting_clf.fit(X_train, y_train)
preds = voting_clf.predict(X_val)
acc = accuracy_score(y_val, preds)
l_loss = log_loss(y_val, preds)
f1 = f1_score(y_val, preds)

print("Accuracy is: " + str(acc))
print("Log Loss is: " + str(l_loss))
print("F1 Score is: " + str(f1))
```

