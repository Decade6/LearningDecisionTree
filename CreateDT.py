## Use Python 3.10


#pip install pandas
#pip install xlrd
# python.exe -m pip install -U scikit-learn
#pip3 install graphviz
#pip3 install pydotplus

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

# Export/Print a decision tree in DOT format.
print(tree.export_graphviz(clf_train, None))

#Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(data.columns.values),
                                class_names=['not survived', 'survived'], rounded=True, filled=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
#Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Create PDF
graph.write_pdf("DT.pdf")

