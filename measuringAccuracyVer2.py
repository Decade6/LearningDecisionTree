from sklearn import tree #For our Decision Tree
import pandas as pd # For our DataFrame
import pydotplus # To create our Decision Tree Graph

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#pip install matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import graph

file = r"titanic.xls"
df = pd.read_excel(file)
print(df.head())
print("name of columns",df.columns)
print(df.head(10))

def age_range(x):
   if x <= 13:
       return 'child'
   elif x <= 18:
       return 'teen'
   elif x  <= 30:
       return 'young adult'
   elif x <= 50:
       return 'adult'
   else:
       return 'older adult'

df['AgeRange'] = df['age'].map(lambda x: age_range(x))


###################################
## 1) Using Sample pop
#Now I will convert the categorical variables into dummy/indicator variables or (binary variables) essentially 1’s and 0's.
#data = pd.get_dummies(df[ ['sex','pclass','parch'] ])
data = pd.get_dummies(df[ ['sex','pclass','parch','sibsp','AgeRange'] ])

#print the new dummy data
print(data)

# The decision tree classifier.
clf = tree.DecisionTreeClassifier()

#split data into test and training set
x = data   # Second column until the last column
y = df['survived']    # First column (Survived) is our target


#this function randomly split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)
#test_size=.3 means that our test set will be 30% of the train set.

# Training the Decision Tree
clf_train = clf.fit(x_train, y_train)
data = pd.get_dummies(df[ ['sex','pclass', 'AgeRange','parch', 'sibsp', 'embarked'] ])
#print the new dummy data
#print(data)

#drop any rows with misssing values
data = data.dropna()
#print the new dummy data
#print(data)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.35, random_state=100)

#######################################
# graph measuring accuracy based upon depth of decision tree
max_depth = []
entropy = []
for i in range(1,10):
 dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
 dtree.fit(x_train, y_train)
 pred = dtree.predict(x_test)
 entropy.append(accuracy_score(y_test, pred))
 ####
 max_depth.append(i)

#plot graph
d = pd.DataFrame({
 'entropy':pd.Series(entropy),
 'max_depth':pd.Series(max_depth)})

plt.plot('max_depth','entropy', data=d, label='entropy')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.legend()
plt.show()
