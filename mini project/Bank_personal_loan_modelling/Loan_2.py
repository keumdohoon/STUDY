import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#error fixed by putting in header and indexcol as 0

import os
# print(os.listdir("./input"))

# original = pd.read_excel('./Bank_personal_loan_modelling.xlsx',"Data")
data = pd.read_csv('./data/csv/Bank_personal_loan_modelling - Clean Data (1).csv',
                            index_col = None,
                            header=0,
                            sep=',',
                            encoding='CP949')
print('data',data.shape)

feature=data.drop("Personal Loan",axis=1)
target=data["Personal Loan"]
loans = feature.join(target)

print(loans)
print(loans.head(5))
print(loans.tail(5))
print(loans.shape)

listItem = []
for col in loans.columns :
    listItem.append([col,loans[col].dtype,
                     loans[col].isna().sum(),
                     round((loans[col].isna().sum()/len(loans[col])) * 100,2),
                    loans[col].nunique(),
                     list(loans[col].sample(5).drop_duplicates().values)]);

dfDesc = pd.DataFrame(columns=['dataFeatures', 'dataType', 'null', 'nullPct', 'unique', 'uniqueSample'],
                     data=listItem)
print(dfDesc)
 #데이터 타입과 null과nullpct,(null 인 값이 몇개인지, ),unique, 와 unique sample을 나타내어준다
 #           dataFeatures dataType  ...  unique                         uniqueSample        
 # 0                  Age    int64  ...      40                 [59, 27, 58, 63, 50]        
 # 1           Experience    int64  ...      42                  [39, 24, 31, 40, 9]        
 # 2               Income    int64  ...     102            [154, 172, 158, 148, 122]        
 # 3             ZIP Code    int64  ...     238  [93305, 94709, 94542, 94309, 90254]        
 # 4               Family    int64  ...       4                            [3, 2, 4]        
 # 5                CCAvg  float64  ...      95                 [0.5, 5.4, 7.2, 3.3]        
 # 6            Education    int64  ...       3                               [2, 3]        
 # 7             Mortgage    int64  ...     141                        [0, 255, 294]        
 # 8   Securities Account    int64  ...       2                                  [0]        
 # 9           CD Account    int64  ...       2                               [1, 0]        
 # 10              Online    int64  ...       2                               [1, 0]        
 # 11          CreditCard    int64  ...       2                               [0, 1]        
 # 12       Personal Loan    int64  ...       1                                  [1]        


# Missing value visualization 비어 있는 값들을 이렇게 히트맵으로 보여준다.
sns.heatmap(loans.isna(),yticklabels=False,cbar=False,cmap='viridis')


loans.describe().transpose()
print('loans',loans)

#이상치를 비주얼화 해준다. 
outvis = loans.copy()
def fungsi(x):
    if x<0:
        return np.NaN
    else:
        return x
    
outvis["Experience"] = outvis["Experience"].apply(fungsi)

sns.heatmap(outvis.isnull(),yticklabels=False,cbar=False,cmap='plasma')



#데이터에 있는 loans를 education을 기준으로 그룹바이 시켜주고 experience의 평균만을 가져와서 비교해준다. 
pd.DataFrame(loans.groupby("Education").mean()["Experience"])




#데이터에 있는 loans를 age을 기준으로 그룹바이 시켜주고 experience의 평균의 마지막 8개를 가져와서 비교해준다. 
pd.DataFrame(loans.groupby("Age").mean()["Experience"]).tail(8)
#age와 exp를 가지고 플롯을 그려준다
pltdf = pd.DataFrame(loans.groupby("Age").mean()["Experience"]).reset_index()
sns.lmplot(x='Age',y='Experience',data=pltdf)
plt.ylabel("Experience (Average)")
plt.title("Average of Experience by Age")
plt.show()

pd.DataFrame(loans[loans["Experience"]<0][["Age","Experience"]].sort_values("Age")).head()

pd.DataFrame(loans[loans["Experience"]<0][["Age","Experience"]].sort_values("Age"))["Experience"].unique()
#feature1에서 결측치를 핸들링하는 방법
loans["Experience"] = loans["Experience"].apply(abs)

loans.describe().transpose()

# Data type analysis
# 각각의 feature가 알맞는 데이터형을 가지고 있는게 매우 중요하다.각각의 feature가 categorical 이나numerical인지에 따라서 바꿔준다. 
# categorical feature into 'int64', and
# numerical feature into 'float64'

print(loans.info())
  #   Column              Non-Null Count  Dtype
  # ---  ------              --------------  -----
  #  0   Age                 480 non-null    int64
  #  1   Experience          480 non-null    int64
  #  2   Income              480 non-null    int64
  #  3   ZIP Code            480 non-null    int64
  #  4   Family              480 non-null    int64
  #  5   CCAvg               480 non-null    float64
  #  6   Education           480 non-null    int64
  #  7   Mortgage            480 non-null    int64
  #  8   Securities Account  480 non-null    int64
  #  9   CD Account          480 non-null    int64
  #  10  Online              480 non-null    int64
  #  11  CreditCard          480 non-null    int64
  #  12  Personal Loan       480 non-null    int64
  # dtypes: float64(1), int64(12)
  #현재는 이런식으로 int타입이 12개 이고 float형식이 1개인데 우리는 이를 다 float형식으로 바꾸어 줄 것이다 
loans[["Age","Experience","Income","CCAvg","Mortgage"]] = loans[["Age","Experience","Income","CCAvg","Mortgage"]].astype(float)
print(loans.info())
   #   Column              Non-Null Count  Dtype
 #   ---  ------              --------------  -----
 #    0   Age                 480 non-null    float64
 #    1   Experience          480 non-null    float64
 #    2   Income              480 non-null    float64
 #    3   ZIP Code            480 non-null    int64
 #    4   Family              480 non-null    int64
 #    5   CCAvg               480 non-null    float64
 #    6   Education           480 non-null    int64
 #    7   Mortgage            480 non-null    float64
 #    8   Securities Account  480 non-null    int64
 #    9   CD Account          480 non-null    int64
 #    10  Online              480 non-null    int64
 #    11  CreditCard          480 non-null    int64
 #    12  Personal Loan       480 non-null    int64
 #   dtypes: float64(5), int64(8)
#위처럼 우리가 설정해둔 가ㅏㅄ들이 float형식으로 바뀌게 된다. 

#multi correlation을 찾아주는데 이는 우리가 회구 모델을 사용하였을때 얼마나 우리의 값을 잘 찾아주느냐를 평가하는 지표이다. 
feature = loans.drop(["ID","Personal Loan"],axis=1)
target = loans["Personal Loan"]

# 1. Heatmap correlation
# Heatmap is one of simplest method to analyze feature correlation.

# 1. Heatmap correlation with only features (X) - we need to know correlation between features and avoid multi-correlation features,
# 2. Heatmap correlation with features (X) and target (y) - we need to know which features that have good correlation with our target,
plt.figure(figsize=(10, 10))
sns.heatmap(feature.corr(),annot=True,square=True)
corr = feature.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(10, 10))
print(sns.heatmap(corr, mask=mask,annot=True,square=True))
plt.show()




# plt.figure(figsize=(10, 10))
# sns.heatmap(feature.join(target).corr(),annot=True,square=True)

# plt.figure(figsize=(20, 20))
# sns.pairplot(feature.join(target).drop(["ZIP Code"],axis=1),hue="Personal Loan")


loans_corr = feature.join(target).corr()

mask = np.zeros((13,13))
mask[:12,:]=1

plt.figure(figsize=(10, 10))
with sns.axes_style("white"):
    sns.heatmap(loans_corr, annot=True,square=True,mask=mask)
plt.show()

sns.distplot(feature["Mortgage"])
plt.title("Mortgage Distribution with KDE")
plt.show()

# Irregular value handling feature 2 (extreme positive skewed data)
SingleLog_y = np.log1p(feature["Mortgage"])              
sns.distplot(SingleLog_y, color ="r")
plt.title("Mortgage Distribution with KDE First Transformation")
plt.show()


DoubleLog_y = np.log1p(SingleLog_y)
sns.distplot(DoubleLog_y, color ="g")
plt.title("Mortgage Distribution with KDE Second Transformation")
plt.show()

loans["Mortgage"] = DoubleLog_y

source_counts =pd.DataFrame(loans["Personal Loan"].value_counts()).reset_index()
source_counts.columns =["Labels","Personal Loan"]
source_counts

#####################################################################################
# fig1, ax1 = plt.subplots()
# explode = (0, 0.15)
# ax1.pie(source_counts["Personal Loan"], explode= explode, labels=source_counts["Labels"], autopct='%1.1f%%',
#          shadow=True, startangle=70)

# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.title("Personal Loan Percentage")
# plt.show()


plt.figure(figsize=(10,6))
sns.distplot(loans[loans["Personal Loan"] == 0]['Income'], color = 'r',label='Personal Loan=0',kde=False)
sns.distplot(loans[loans["Personal Loan"] == 1]['Income'], color = 'b',label='Personal Loan=1',kde=False)
plt.legend()
plt.title("Income Distribution")
plt.show()

plt.figure(figsize=(10,6))
sns.distplot(loans[loans["Personal Loan"] == 0]['CCAvg'], color = 'r',label='Personal Loan=0',kde=False)
sns.distplot(loans[loans["Personal Loan"] == 1]['CCAvg'], color = 'b',label='Personal Loan=1',kde=False)
plt.legend()
plt.title("CCAvg Distribution")
plt.show()



plt.figure(figsize=(10,6))
sns.distplot(loans[loans["Personal Loan"] == 0]['Experience'], color = 'r',label='Personal Loan=0',kde=False)
sns.distplot(loans[loans["Personal Loan"] == 1]['Experience'], color = 'b',label='Personal Loan=1',kde=False)
plt.legend()
plt.title("Experience Distribution")
plt.show()


sns.countplot(x='Securities Account',data=loans,hue='Personal Loan')
plt.title("Securities Account Countplot")
plt.show()

sns.countplot(x='Family',data=loans,hue='Personal Loan')
plt.title("Family Countplot")
plt.show()


sns.boxplot(x='Education',data=loans,hue='Personal Loan',y='Income')
plt.legend(loc='lower right')
plt.title("Education and Income Boxplot")
plt.show()

sns.boxplot(x='Family',data=loans,hue='Personal Loan',y='Income')
plt.legend(loc='upper center')
plt.title("Family and Income Boxplot")
plt.show()


feature = loans.drop(["ID","Personal Loan"],axis=1)
target = loans["Personal Loan"]

#features removing
feature = feature.drop(["ZIP Code","Age","Online","CreditCard"],axis=1)

feature["Combination"] = (feature["Income"]/12)**feature["CCAvg"]

from sklearn.preprocessing import MinMaxScaler,StandardScaler,robust_scale
scaler = StandardScaler();



colscal=["Experience","Mortgage","Income","CCAvg","Combination"]

scaler.fit(feature[colscal])
scaled_features = pd.DataFrame(scaler.transform(feature[colscal]),columns=colscal)

feature =feature.drop(colscal,axis=1)
feature = scaled_features.join(feature)
#modelling
from sklearn.model_selection import train_test_split,cross_val_score
X_train, X_test, y_train, y_test = train_test_split(feature,target,
                                                    test_size=0.20,
                                                    random_state=101)

y_train.value_counts()
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=101)
xgb.fit(X_train, y_train)

from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score, confusion_matrix, accuracy_score,recall_score

predict = xgb.predict(X_test)
predictProb = xgb.predict_proba(X_test)
print(predictProb)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
# score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")
print(score1)

print("confusion_matrix :\n",confusion_matrix(y_test, predict))
print("\nclassification_report :\n",classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
# print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
print("\nCross Validation Recall :",score1.mean())
# print("Cross Validation Roc Auc :",score2.mean())
########################################

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=101)

X_ros, y_ros = ros.fit_sample(X_train, y_train)

pd.Series(y_ros).value_counts()

xgb = XGBClassifier(n_estimators=97,random_state=101)
xgb.fit(X_ros, y_ros)

predict = xgb.predict(X_train.values)
predictProb = xgb.predict_proba(X_train.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print("confusion_matrix :\n",confusion_matrix(y_train, predict))
print("\nclassification_report :\n",classification_report(y_train, predict))
print('Recall Score',recall_score(y_train, predict))
print('ROC AUC :', roc_auc_score(y_train, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_train, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_train, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())

#XGBClassifier model with balance dataset (test evaluation)
predict = xgb.predict(X_test.values)
predictProb = xgb.predict_proba(X_test.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print("confusion_matrix :\n",confusion_matrix(y_test, predict))
print("\nclassification_report :\n",classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())


from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=xgb,
                                                       X=feature,
                                                       y=target,
                                                       train_sizes=np.linspace(0.01, 1.0, 10),
                                                       cv=10)

print(train_scores)
# Mean value of accuracy against training data
train_mean = np.mean(train_scores, axis=1)
print(train_mean)
print(train_sizes)
# Standard deviation of training accuracy per number of training samples
train_std = np.std(train_scores, axis=1)

# Same as above for test data
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot training accuracies 
plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')
# Plot the variance of training accuracies
plt.fill_between(train_sizes,
                train_mean + train_std,
                train_mean - train_std,
                alpha=0.15, color='red')

# Plot for test data as training data
plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 
        label='Test Accuracy')
plt.fill_between(train_sizes,
                test_mean + test_std,
                test_mean - test_std,
                alpha=0.15, color='blue')

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Train and Test Accuracy Comparison")
plt.show()

coef1 = pd.Series(xgb.feature_importances_,feature.columns).sort_values(ascending=False)

pd.DataFrame(coef1,columns=["Features"]).transpose().plot(kind="bar",title="Feature Importances") #for the legends

coef1.plot(kind="bar",title="Feature Importances")