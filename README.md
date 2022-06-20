# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the necessary packages using import statement.
2. Read the given csv file and print the number of contents to be displayed.
3. Split the dataset using train_test_split.
4.Calculate Y_Pred and accuracy.
5. Display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Sai sonica CH
RegisterNumber: 212219040130
*/
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

## Data.head():
![image](https://user-images.githubusercontent.com/79306169/174664276-b63f4010-8453-4611-8751-b766ba84a60c.png)
## Data.info():
![image](https://user-images.githubusercontent.com/79306169/174664301-bf415b40-2897-45c8-8e12-ee0792dfef5f.png)
## Data.isnull().sum():
![image](https://user-images.githubusercontent.com/79306169/174664325-edd5bb33-7625-4bb6-a96f-69ba4c4ac0d4.png)
## Y_Pred:
![image](https://user-images.githubusercontent.com/79306169/174664340-29640db9-3e75-4fa1-ab8e-1cc619282449.png)
## Accuracy:
![image](https://user-images.githubusercontent.com/79306169/174664370-97053896-6848-4dac-810e-72df037aeee3.png)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
