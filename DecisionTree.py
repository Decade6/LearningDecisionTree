"""

Name:Declan Murphy
Date:04/15/2024
Assignment:Module 12: Project -Learning Decision Tree
Due Date:04/14/2024
About this project: Create a graph that measures accuracy based upon training set size,
a subset of the data using 4 discrete domain attributes and the attribute,
and a graph that measures accuracy based upon the depth of the decision tree.
Assumptions:StarWars dataset is valid
All work below was performed by Declan Murphy

"""


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
from sklearn.tree import DecisionTreeClassifier, export_graphviz

#pip install matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import graph
# Load the dataset
df = pd.read_csv("StarWars.csv", encoding='ISO-8859-1')

# Preprocessing: Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(df[['Gender', 'Age', 'Household Income', 'Education']])
target = df['Which character shot first?'].map({"Han": 1, "Greedo": 0, "I don't understand this question": 2}).fillna(2)

# 1. Graph measuring accuracy based upon training set size
training_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
accuracies = []

for size in training_sizes:
    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=size, random_state=42)
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)

plt.figure()
plt.plot(training_sizes, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
plt.title('Accuracy vs Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_set_size_accuracy.png')
plt.show()

# 2 & 3. Creating a subset and Decision Tree using entropy
X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.5, random_state=42)  # Example split
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

# Export/Print a decision tree in DOT format.
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=list(data.columns.values),
                           class_names=['Greedo', 'Han', 'Unknown'],  # Adjust class names as necessary
                           rounded=True, filled=True)

# Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Save the graph to a PDF file
graph.write_pdf("decision_tree.pdf")

# 4. Graph measuring accuracy based upon depth of decision tree
max_depths = range(1, 10)
depth_accuracies = []

for depth in max_depths:
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    depth_accuracies.append(accuracy)

plt.figure()
plt.plot(max_depths, depth_accuracies, marker='o', linestyle='-', color='r', label='Accuracy')
plt.title('Accuracy vs Depth of Decision Tree')
plt.xlabel('Depth of Decision Tree')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('depth_accuracy.png')
plt.show()
