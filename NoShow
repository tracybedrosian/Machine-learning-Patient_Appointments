### import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### read in data and examine
df = pd.read_csv('NoShow.csv')
df.dtypes
df.columns
df.describe()
df['Status'].describe()

df['index_col'] = df.index # add a person index
df = df.drop(['AppointmentRegistration', 'ApointmentData'], axis=1) # dropping this for now - awaiting time captures some of this

### visually explore data
sns.distplot(df['Age']) # fairly even representation of all ages
sns.distplot(df['AwaitingTime']) # some outliers here to think about

sns.barplot(x='Status', y='Age', data=df) # no shows tend to be younger
sns.barplot(x='Status', y='AwaitingTime', data=df) # no shows booked earlier

sns.regplot(x='Age', y='AwaitingTime', data=df) # babies and elderly have standing appts

### check missing data and clean up data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(15)

df.Handcap[df.Handcap > 1] = 1 # recode false values
df.Sms_Reminder[df.Sms_Reminder > 1] = 1 # recode false values

dums = pd.get_dummies(df['Gender']) # encode gender as binary integer
dums.columns = ['Female','Male']
dums.drop(['Female'], axis=1, inplace=True)

dumst = pd.get_dummies(df['Status']) # encode status as binary integer
dumst.columns = ['No-Show','Show-Up']
dumst.drop(['Show-Up'], axis=1, inplace=True)

dumd = pd.get_dummies(df['DayOfTheWeek']) # encode day as binary integer
dumd.columns = ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

df = df.join(dums)
df = df.join(dumd)
df = df.join(dumst)
df.drop(['Gender', 'DayOfTheWeek', 'Status'], axis=1, inplace=True)

### split df into train and test
from sklearn.model_selection import train_test_split

X_all = df.drop(['No-Show', 'index_col'], axis=1)
y_all = df['No-Show']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

### train a model
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test) # make predictions

random_forest.score(X_train, y_train) # check accuracy on training set
random_forest.score(X_test, y_test) # check accuracy on test set since we have the answers

### plot feature importance
importances = random_forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in random_forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices) # age is most important followed by sms_reminders

