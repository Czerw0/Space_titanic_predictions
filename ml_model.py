from space_titanic_train_fe import X_transformed, y
import numpy as np
import matplotlib.pyplot as plt

#sklearn 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV
#models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=0)

# Logistic Regression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)

y_pred_prop = clf.predict_proba(X_test)

print("CLF SCORE: ", clf.score(X_test, y_test))

#print coefficients matrix
print("Coefficients: ", clf.coef_)
print("Intercept: ", clf.intercept_)
print("\n")
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred),display_labels=clf.classes_)
disp.plot()
plt.show()
print("\n")

#Recall, Precision, F1
print("Classification Report: ")
print("Recall score: ", recall_score(y_test,y_pred))
print("Precision score: ", precision_score(y_test,y_pred))
print("F1 score: ", f1_score(y_test,y_pred))

#ROC curve
svc_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.show()

#Kfold cross validation
scores = cross_val_score(clf, X_train, y_train, cv=10,scoring = 'precision')
print(scores)
print(scores.mean())
print(scores.std())
print("\n")

#Grid search 1
grid = [
    {"C": np.logspace(-3, 3, 7), "penalty": ["l2"], "solver": ["lbfgs"], "class_weight": ['balanced', None]},
    {"C": np.logspace(-3, 3, 7), "penalty": ["l1"], "solver": ["saga"], "class_weight": ['balanced', None]}
]

# Initialize GridSearchCV
logreg = LogisticRegression(max_iter=5000)
logreg_cv = GridSearchCV(logreg, grid, cv=10, scoring="f1", n_jobs=-1, error_score="raise")

# Fit GridSearch
logreg_cv.fit(X_train, y_train)
print("\n")
# Print results
print("Best Parameters:", logreg_cv.best_params_)
print("Best F1 Score:", logreg_cv.best_score_)

np.logspace(-3,3,7)

#Different models 

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
rfc = RandomForestClassifier(random_state=0).fit(X_train, y_train)
svc = SVC(random_state=0).fit(X_train, y_train)
gnb = GaussianNB().fit(X_train, y_train)
knn = KNeighborsClassifier().fit(X_train, y_train)
dtr = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)

y_pred_clf = clf.predict(X_test)
y_pred_rfc = rfc.predict(X_test)
y_pred_svc = svc.predict(X_test)
y_pred_gnb = gnb.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_dtr = clf.predict(X_test)
print("\n")
print("LogisticRegression")
print("Recall score: ", recall_score(y_test,y_pred))
print("Precision score: ", precision_score(y_test,y_pred))
print("F1 score: ", f1_score(y_test,y_pred))
print("\n")
print("RandomForestClassifier")
print("Recall score: ", recall_score(y_test,y_pred_rfc))
print("Precision score: ", precision_score(y_test,y_pred_rfc))
print("F1 score: ", f1_score(y_test,y_pred_rfc))
print("\n")
print("SVC")
print("Recall score: ", recall_score(y_test,y_pred_svc))
print("Precision score: ", precision_score(y_test,y_pred_svc))
print("F1 score: ", f1_score(y_test,y_pred_svc))
print("\n")
print("GaussianNB")
print("Recall score: ", recall_score(y_test,y_pred_gnb))
print("Precision score: ", precision_score(y_test,y_pred_gnb))
print("F1 score: ", f1_score(y_test,y_pred_gnb))
print("\n")
print("KNeighborsClassifier")
print("Recall score: ", recall_score(y_test,y_pred_knn))
print("Precision score: ", precision_score(y_test,y_pred_knn))
print("F1 score: ", f1_score(y_test,y_pred_knn))
print("\n")
print("Decision Tree")
print("Recall score: ", recall_score(y_test,y_pred_dtr))
print("Precision score: ", precision_score(y_test,y_pred_dtr))
print("F1 score: ", f1_score(y_test,y_pred_dtr))
print("\n")
rfc_cv = RandomForestClassifier()
svc_cv = SVC()
gnb_cv = GaussianNB()
knn_cv = KNeighborsClassifier()
logr_cv = LogisticRegression()
dtr_cv = DecisionTreeClassifier()

models = [rfc_cv,svc_cv,gnb_cv,knn_cv,logr_cv, dtr_cv]
print("Cross validation scores: ")
for i in models:
  scores = cross_val_score(i, X_train, y_train, cv=10)
  print(i,": with mean ",scores.mean()," and std ",scores.std())

#chosing best model - RandomForestClassifier

#Grid search 2
grid = {
    #'bootstrap': [True],
    'max_depth': [5,8,10],
    'max_features': ['sqrt', 'log2',None,5],
    #'min_samples_leaf': [3, 4, 5],
    #'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200,500]
}
rfc_grid = RandomForestClassifier()
rfc_grid_cv=GridSearchCV(rfc_grid,grid,cv=10)
rfc_grid_cv.fit(X_train,y_train)

print("\n")
print("Random Forest Classifier:")
print("tuned hpyerparameters :(best parameters) ",rfc_grid_cv.best_params_)
print("accuracy:",rfc_grid_cv.best_score_)

print("\n")
print(rfc_grid_cv.cv_results_)

#Final model
rfc_grid_finalv1 = RandomForestClassifier(max_depth=5, max_features = 'log2', n_estimators = 500,random_state = 3)
rfc_grid_finalv2 = RandomForestClassifier(max_depth=8, max_features = 'sqrt', n_estimators = 100,random_state = 3)
rfc_grid_finalv3 = RandomForestClassifier(max_depth=8, n_estimators=500, random_state=3,criterion = 'entropy')
rfc_grid_finalv4 = RandomForestClassifier(max_depth=10, max_features = 'log2', n_estimators = 500,random_state = 3)
rfc_grid_final_vanilla = RandomForestClassifier(random_state=3)

models = [rfc_grid_finalv1, rfc_grid_finalv2,rfc_grid_finalv3,rfc_grid_finalv4,rfc_grid_final_vanilla]
print("\n")
for i in models:
  scores = cross_val_score(i, X_train, y_train, cv=10)
  print(i,": with mean ",scores.mean()," and std ",scores.std())
print("\n")
best_rf = RandomForestClassifier(max_depth=10, max_features='log2', n_estimators=500, random_state=3)
best_rf.fit(X_train, y_train)