import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class WESADLDA:
    def __init__(self):
        pass
    def transform(self, samples):
        X = samples.iloc[:, samples.columns != 'label']
        y = samples.iloc[:, samples.columns == 'label'].values.ravel()  # To convert y into a 1D array
        X.columns = X.columns.astype(str)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        classifier = RandomForestClassifier(max_depth=4, random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print('Accuracy' + str(accuracy_score(y_test, y_pred)))
        return X_train, X_test, y_train, y_test



