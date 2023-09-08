import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# Read in Data From CSV
data = pd.read_csv('cancer_data.csv')

# Keep copy of origional data set
origional_data = data


# Preliminary data type checking for proceeding datatype conversion
print("\n")
print("INITIAL DATA DESCRIPTION:")
print("\n")
print(data.info())
print("\n")


# Change target str type values to type int
data['diagnosis'].replace({'M': 1,'B': 0}, inplace=True)


# Remove Id column for correlation observation and pre processing
data = data.drop(columns=['id'])
data_test = data


# Corrrelation coefficient in order to determine correlation dynamic between features
target_corr = data.corr()['diagnosis']
print("INITAL TARGET CORRELATION:")
print("\n")
print(target_corr)


# Corrrelation coefficient in order to decide which features are poorly correlated and which features are heavily correlated with eachother
correlation_matrix = data.corr()
print("\n")
print('CORRELATION MATRIX:')
print("\n")
print(correlation_matrix)

matrix_mask = np.triu(data.corr())
sns.heatmap(correlation_matrix, cbar=False, annot=True, mask= matrix_mask )
plt.show()


# Remove poorly correlated Data
to_remove = []
for name in range(len(target_corr)):
    if target_corr.iloc[name] < 0.4:
        to_remove.append(target_corr.index[name])


new_target_corr = target_corr.drop(to_remove)
print("\n")
print("UPDATED TARGET CORRELATION:")
print("\n")
print(new_target_corr)
print(len(new_target_corr))

new_data = data.drop(columns=to_remove)
data = new_data
correlation_matrix = data.corr()
print('\n')
print('DETERMINE HIGHEST CORRELATION AMONG FEATURES:')

i = 0
j = 0
to_remove2 = []
for column in correlation_matrix[j:]:
    col = correlation_matrix[column].to_numpy()
    for value in col[j:]:
        if (value > 0.97) & (value != 1):
            name = correlation_matrix.iloc[i].name
            print('\n')
            print('FEATURE 1:',column, '\nCORRELATION WITH TARGET:',target_corr.iloc[j])
            print('FEATURE 2:',name, '\nCORRELATION WITH TARGET:',target_corr.iloc[i])
            print('CORRELATION VALUE:',value )
            if (target_corr.iloc[j] > target_corr.iloc[i]) & (target_corr.index[j] not in to_remove2):
                if target_corr.index[j] not in to_remove2:
                    to_remove2.append(target_corr.index[j])
                    # to_remove2.append(target_corr.iloc[j])
            elif (target_corr.iloc[i] > target_corr.iloc[j]) & (target_corr.index[i] not in to_remove2):
                if target_corr.index[i] not in to_remove2:
                    to_remove2.append(target_corr.index[i])
                    # to_remove2.append(target_corr.iloc[i])
        i+=1
    i=0
    j+=1
print('\n')
print('HIGHLY CORRELATED FEATURES:',to_remove2)


# Remove parameter with lowest correlation to target
min = 1
for x in range(len(to_remove2)):
    if target_corr[to_remove2[x]] < min:
        min = target_corr[to_remove2[x]]
        name = to_remove2[x]
print('\n')
print('FEATURE TO BE REMOVED:',name, '\nFEATURE CORRELATION TO TARGET:', min)
new_data = data.drop(columns=name)
data = new_data


# Final Data Set Before Model Training and Testing
print('\n')
print("FINAL ADJUSTED DATASET:")
print('\n')
print(data)


# data = data.drop(columns=['texture_mean'])
print('\n')
print('FINAL TARGET CORRELATIONS:')
print('\n')
print(data.corr()['diagnosis'])


# Store target variable in series for model implementation
diagnosisVar = data['diagnosis']

# Remove dependent Variable 'diagnosis' from data set
data = data.drop(columns=['diagnosis'])


# Data standerdization for model training
scaled_data = StandardScaler().fit_transform(data)
scaled = pd.DataFrame(data= pd.DataFrame(scaled_data).values, columns=data.columns)

print('\n')
print("FINAL SCALED DATA:")
print('\n')
print(scaled)


# Split into training and testing
x = scaled[scaled.columns].values
y = diagnosisVar.values

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.60, random_state=5)


# Regression Model Implementation

# Find line of best fit using Logistic Regression according to training data
model = LogisticRegression().fit(x_train, y_train)

#Apply Model to test data
target_prediction = model.predict(x_test)


# Model Accuracy Trainig and Testing Results
print("\n")
print("Breast Cancer Classificaiton Model Using: Leaner Regression")
print("\n")
print("Accuracy")
print("Training:",round(accuracy_score(y_train, model.predict(x_train))*100,2),'%')
print("Testing:",round(accuracy_score(y_test, model.predict(x_test))*100,2),'%')
print("\n")
print("Precision")
print("Training:",round(precision_score(y_train, model.predict(x_train))*100,2),"%")
print("Testing:",precision_score(y_test, model.predict(x_test))*100,"%")
print("\n")
print("Recall")
print("Training:",round(recall_score(y_train, model.predict(x_train))*100,2),"%")
print("Testing:",round(recall_score(y_test, model.predict(x_test))*100,2),"%")
print("\n")
cm = confusion_matrix(y_test, target_prediction)
print("Confusion Matrix\n",cm)

cm_heatmap = sns.heatmap(confusion_matrix(y_test, target_prediction),annot=True, cbar=False, fmt= "d" )
plt.title('Logistic Regression Model Confusion Matrix')
plt.xlabel('Model Prediction')
plt.ylabel('Actual Value')
plt.show()



# print('X:',x)
# print(len(x))
# print('Y:',y)
# print(len(y))

# print("x train")
# print(x_train)
# print(len(x_train))

# print("y train")
# print(y_train)
# print(len(y_train))

# print("x test")
# print(x_test)
# print(len(x_test))

# print("y test")
# print(y_test)
# print(len(y_test))