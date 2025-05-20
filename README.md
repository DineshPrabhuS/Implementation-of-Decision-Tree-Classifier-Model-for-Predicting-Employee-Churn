# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.

## Program:
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: DINESH PRABHU S

RegisterNumber: 212224040077
*/
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:", accuracy)

dt.predict([[0.5,0.8,9,260, 6,0,1,2]])
```

## Output:
![image](https://github.com/user-attachments/assets/eb9d9146-db10-432e-8020-6436667017e0)

![image](https://github.com/user-attachments/assets/b1e6ae35-25fd-4e96-b81d-9f52301adf1c)

![image](https://github.com/user-attachments/assets/5fffbff9-eaed-4026-9938-ae9f41d9640d)

![image](https://github.com/user-attachments/assets/62ede38b-4289-4854-8a95-4d8995082dad)

![image](https://github.com/user-attachments/assets/ccac0799-994d-43ec-9808-6551385af5b6)

![image](https://github.com/user-attachments/assets/b3cf4969-2e9e-4ecb-a284-e28844b42544)

![image](https://github.com/user-attachments/assets/44cf1c2b-5751-43bf-a592-834d4954b519)

![image](https://github.com/user-attachments/assets/02e05652-4d3c-40f8-8a18-8a2f1b82337f)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
