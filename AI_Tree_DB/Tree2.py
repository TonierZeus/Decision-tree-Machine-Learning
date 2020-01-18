import pandas as pd #imort pandas
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split
from sklearn import metrics #import metric do obliczeń

#nazwanie kolumn
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

#import bazy z pliku
dataset = pd.read_csv("diabetesV2.csv", header=None, names=col_names)

#wyświetlenie bazy
print(dataset)

#podzielenie bazy na obiekty i cechy
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = dataset[feature_cols] #cechy
y = dataset.label #obiekty

#podział bazy na grupy testujące 30% i trenujące 70%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#stworzenie klasyfikatora drzewa
clf = DecisionTreeClassifier()

#testowanie klasyfikatora drzewa
clf = clf.fit(X_train,y_train)

#przewidywanie wyniku
y_pred = clf.predict(X_test)

#wyświetlenie jak często model klasyfikacji jest poprawny(0.67)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#import graphviz
from sklearn.tree import export_graphviz

#eskport drzewa w postaci txt
export_graphviz(clf,out_file ='tree.txt')

#optymalicazcja drzewa
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#wyświetlenie jak czesto model klasyikacji jest poprawan po optymalizacji(0.77)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#eksport po optymalizacji
export_graphviz(clf,out_file ='treeOP.txt')
#http://www.webgraphviz.com/ wyświetlenie grafów