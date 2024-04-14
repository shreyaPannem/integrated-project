import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
!pip install hvplot
import hvplot.pandas
import os
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")
from google.colab import files


uploaded = files.upload()
df = pd.read_csv('diabetes12.csv')
df.head()
df.tail()
df.shape
df.columns
categorical_val = []
continous_val = []
for column in df.columns:
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)
        categorical_val
        df.describe()
        df.isnull().sum()
        print(df.info())
        plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True, cmap='terrain')
plt.figure(figsize=(20,10))
sns.set_context('notebook',font_scale = 1.5)
sns.barplot(x=df.age.value_counts()[:10].index,y=df.age.value_counts()[:10].values)
plt.tight_layout()
plt.figure(figsize=(15, 15))

for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    df[df["diabetes"] == 0][column].hist(bins=35, color='green', label='Have Heart Disease = NO', alpha=0.6)
    df[df["diabetes"] == 1][column].hist(bins=35, color='blue', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)
    plt.figure(figsize=(15, 15))

for i, column in enumerate(continous_val, 1):
    plt.subplot(3, 2, i)
    df[df["diabetes"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    df[df["diabetes"] == 1][column].hist(bins=35, color='green', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)
    sns.pairplot(data=df)
    df.hist(figsize=(12,12), layout=(5,3));
df.plot(kind='box', subplots=True, layout=(5,3), figsize=(12,12))
plt.show()
# Create another figure
plt.figure(figsize=(9, 7))

# Scatter with postivie examples
plt.scatter(df.age[df.diabetes==1],
            df.diabetes[df.diabetes==1],
            c="red")

# Scatter with negative examples
plt.scatter(df.age[df.diabetes==0],
            df.diabetes[df.diabetes==0],
            c="blue")

# Add some helpful info
plt.title("Heart Disease in function of Age and diabetes")
plt.xlabel("Age")
plt.ylabel("diabetes")
plt.legend(["Disease", "No Disease"]);
sns.catplot(data=df, x='bmi', y='age',  hue='diabetes', palette='husl')
sns.barplot(data=df, x='age', y='heart_disease', hue='diabetes', palette='spring')
df['age'].value_counts()
df['diabetes'].value_counts()
df['bmi'].value_counts()
sns.countplot(x='age', data=df, palette='husl', hue='diabetes')
sns.countplot(x='diabetes',palette='BuGn', data=df)
sns.countplot(x='hypertension',hue='diabetes',data=df)
df['hypertension'].value_counts()
sns.countplot(x='HbA1c_level',data=df, hue='diabetes', palette='BuPu' )
sns.countplot(x='blood_glucose_level', hue='age',data=df, palette='terrain')
df['blood_glucose_level'].value_counts()  # chest pain type
sns.countplot(x='blood_glucose_level' ,hue='diabetes', data=df, palette='rocket')
sns.countplot(x='hypertension', hue='gender(male 0 female 1)',data=df, palette='BrBG')
sns.boxplot(x='gender(male 0 female 1)', y='bmi', hue='diabetes', palette='seismic', data=df)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()
columns_to_scale = ['age','bmi','HbA1c_level','blood_glucose_level']
df[columns_to_scale] = StandardScaler.fit_transform(df[columns_to_scale])
X= df.drop(['diabetes'], axis=1)
y= df['diabetes']
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)
print('X_train-', X_train.size)
print('X_test-',X_test.size)
print('y_train-', y_train.size)
print('y_test-', y_test.size)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

model1=lr.fit(X_train,y_train)
prediction1=model1.predict(X_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,prediction1)
cm
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(y_test,prediction1)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()
sns.heatmap(cm, annot=True,cmap='BuPu')
TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]
print('Testing Accuracy:',(TP+TN)/(TP+TN+FN+FP))
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction1)
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction1))
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()
model2=dtc.fit(X_train,y_train)
prediction2=model2.predict(X_test)
cm2= confusion_matrix[y_test,prediction2]
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()
model2=dtc.fit(X_train,y_train)
prediction2=model2.predict(X_test)
cm2= confusion_matrix[y_test,prediction2]
cm2
from sklearn.metrics import confusion_matrix

cm2=confusion_matrix(y_test,prediction2)
cm2
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(y_test,prediction2)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()
sns.heatmap(cm2, annot=True,cmap='BuPu')
accuracy_score(y_test,prediction2)
print(classification_report(y_test, prediction2))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
model3 = rfc.fit(X_train, y_train)
prediction3 = model3.predict(X_test)
confusion_matrix[y_test, prediction3]
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(y_test,prediction3)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()
accuracy_score(y_test, prediction3)
print(classification_report(y_test, prediction3))
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

svm=SVC()
model4=svm.fit(X_train,y_train)
prediction4=model4.predict(X_test)
cm4= confusion_matrix[y_test,prediction4]
cm4
accuracy_score(y_test, prediction4)
print(classification_report(y_test, prediction4))
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
model5 = NB.fit(X_train, y_train)
prediction5 = model5.predict(X_test)
cm5= confusion_matrix[y_test, prediction5]
cm5
accuracy_score(y_test, prediction5)
print(classification_report(y_test, prediction5))
print('cm4', cm4)
print('-----------')
print('cm5',cm5)
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier()
model6 = KNN.fit(X_train, y_train)
prediction6 = model6.predict(X_test)
cm6= confusion_matrix[y_test, prediction6]
cm6
print(classification_report(y_test, prediction6))
print('LOGISTIC REGRESSION :', accuracy_score(y_test, prediction1))
print('DECISION TREE :', accuracy_score(y_test, prediction2))
print('RANDOM FOREST :', accuracy_score(y_test, prediction3))
print('GAUSSIAN NB: ', accuracy_score(y_test, prediction4))
print('SVC :', accuracy_score(y_test, prediction4))
print('KNN :', accuracy_score(y_test, prediction6))