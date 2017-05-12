# Machine-learning-Patient_Appointments
Building a random forest classifier in Python to predict patient no-shows

Medical appointment no shows data was obtained from Kaggle https://www.kaggle.com/joniarroba/noshowappointments

## Import libraries
```import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```
## Read in data and examine
```df = pd.read_csv('NoShow.csv')
df.dtypes
df.columns
df.describe()
df['Status'].describe()

df['index_col'] = df.index # add a person index
df = df.drop(['AppointmentRegistration', 'ApointmentData'], axis=1)
```
This dataset contains patient information like age, gender, and medical conditions, as well as appointment data like the day of the week, whether a text message reminder was sent, and the 'awaiting time' from booking to actual appointment. We're dropping the columns containing the date the appointment was made and the date of the actual appointment for now - the variable called awaiting time already captures some of the important information contained there. We can come back to the actual dates later if desired.

# Visually explore the data
```sns.distplot(df['Age'])
sns.distplot(df['AwaitingTime'])

sns.barplot(x='Status', y='Age', data=df)
sns.barplot(x='Status', y='AwaitingTime', data=df)

sns.regplot(x='Age', y='AwaitingTime', data=df)
```
![alt text](https://github.com/tracybedrosian/Machine-learning-Patient_Appointments/blob/master/NoShow%20Age.png)
![alt text](https://github.com/tracybedrosian/Machine-learning-Patient_Appointments/blob/master/NoShow%20Time.png)

Examining the distribution of age reveals a fairly even representation of all ages in this patient sample. Visualizing the distribution of awaiting time shows that most patients booked an appointment less than a month earlier, but a few patients had scheduled a year in advance. 

![alt text](https://github.com/tracybedrosian/Machine-learning-Patient_Appointments/blob/master/NoShow%20AgeBarPlot.png)
![alt text](https://github.com/tracybedrosian/Machine-learning-Patient_Appointments/blob/master/NoShow%20TimeBarPlot.png)

By comparing the effect of these two variables on attendance, we can see that patients who are no shows tend to be younger and tend to have booked further in advance. 

![alt text](https://github.com/tracybedrosian/Machine-learning-Patient_Appointments/blob/master/NoShow%20AgexTime.png)

Visualizing the relationship between age and awaiting time suggests that groups on the younger or older ends tend to have booked further in advance. This makes sense because infants, young children, and elderly folks may require more regular appointments.

# Check for missing data and clean up data
```total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(15)

df.Handcap[df.Handcap > 1] = 1
df.Sms_Reminder[df.Sms_Reminder > 1] = 1

dums = pd.get_dummies(df['Gender'])
dums.columns = ['Female','Male']
dums.drop(['Female'], axis=1, inplace=True)

dumst = pd.get_dummies(df['Status'])
dumst.columns = ['No-Show','Show-Up']
dumst.drop(['Show-Up'], axis=1, inplace=True)

dumd = pd.get_dummies(df['DayOfTheWeek'])
dumd.columns = ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

df = df.join(dums)
df = df.join(dumd)
df = df.join(dumst)
df.drop(['Gender', 'DayOfTheWeek', 'Status'], axis=1, inplace=True)
```
Fortunately there are no missing values to deal with! But there do seem to be some mis-encoded values. Handicap status and SMS reminder status should be binary values of 0 or 1, but there are a few larger numbers. We can recode any of these apparently false values to 1. We also will encode categorical variables like Gender, Status, and Day as binary integers and join all of this data back together.

# Split data into train and test sets
```from sklearn.model_selection import train_test_split

X_all = df.drop(['No-Show', 'index_col'], axis=1)
y_all = df['No-Show']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)
```
80% of data will be used to train our model and we'll keep 20% for testing.

# Fit a random forest classifier
```from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
random_forest.score(X_test, y_test)
```
This model is about 85% accurate on the training data and 65% accurate on the test data. We could play around with parameters to improve this but I'll leave it for now.

# Plot feature importance
```importances = random_forest.feature_importances_
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
plt.xticks(range(X_train.shape[1]), indices)
```
![alt text](https://github.com/tracybedrosian/Machine-learning-Patient_Appointments/blob/master/NoShow%20Feat%20Imp.png)

Age is the most important predictor of whether someone shows up for their appointment or not, followed by whether they received a text message reminder or not.
