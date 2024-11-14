# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VINOTH M P
RegisterNumber:212223240182
*/
import chardet
with open('spam.csv','rb') as file:
    result = chardet.detect(file.read(10000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()
data.info()
data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

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

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc
```

## Output:
### Encoding:
![image](https://github.com/user-attachments/assets/fae5d304-fe30-43aa-9eba-8255a181fb9f)

### Head():
![image](https://github.com/user-attachments/assets/d824d63e-db25-42b9-823f-7cf6c2a12a78)

### Info():
![image](https://github.com/user-attachments/assets/028ed981-94bf-404b-941b-06098c18339d)

### isnull().sum():
![image](https://github.com/user-attachments/assets/d5f2de14-34de-4a6b-a3db-0a7ee283d451)

### Prediction of y:
![image](https://github.com/user-attachments/assets/c88796f9-99ab-492f-835b-460e7eea06b3)

### Accuracy:
![image](https://github.com/user-attachments/assets/95288e43-ac2a-4515-8adb-f312844cede0)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
