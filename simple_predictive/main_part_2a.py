# implemented from
# https://medium.com/datadriveninvestor/a-simple-guide-to-creating-predictive-models-in-python-part-1-8e3ddc3d7008

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv('Clean_Data.csv')
df.info()
df.head()

feat = df.drop(columns=['Exited'], axis=1)
label = df['Exited']

X_train, X_test, Y_train, Y_test = train_test_split(feat, label, test_size=0.3)

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

support_vector_classifier = SVC(kernel='rbf')
support_vector_classifier.fit(X_train, Y_train)
y_pred_svc = support_vector_classifier.predict(X_test)

cm_support_vector_classifier = confusion_matrix(Y_test, y_pred_svc)

print(cm_support_vector_classifier, '\n\n')

numerator = cm_support_vector_classifier[0][0] + cm_support_vector_classifier[1][1]
denominator = sum(cm_support_vector_classifier[0]) + sum(cm_support_vector_classifier[1])
acc_svc = (numerator/denominator) * 100
print('Accuracy: ', str(acc_svc) + '%')

# cross_val_svc = cross_val_score(estimator=SVC(kernel='rbf'), X=X_train, y=Y_train, cv=10, n_jobs=-1)
# print('Cross Validation Accuracy : ', round(cross_val_svc.mean() * 100, 2), '%')
#
# random_forest_classifier = RandomForestClassifier()
# random_forest_classifier.fit(X_train, Y_train)
# y_pred_rfc = random_forest_classifier.predict(X_test)
#
# cm_random_forest_classifier = confusion_matrix(Y_test, y_pred_rfc)
# print(cm_random_forest_classifier)
#
# numerator = cm_random_forest_classifier[0][0] + cm_random_forest_classifier[1][1]
# denominator = sum(cm_random_forest_classifier[0]) + sum(cm_random_forest_classifier[1])
# acc_rfc = (numerator/denominator) * 100
# print('Accuracy : ', round(acc_rfc, 2), '%')
#
# cross_val_rfc = cross_val_score(estimator=RandomForestClassifier(), X=X_train, y=Y_train, cv=10, n_jobs=-1)
# print('Cross Validation Accuracy : ', round(cross_val_rfc.mean() * 100, 2), '%')
#
# xgb_classifier = XGBClassifier()
# xgb_classifier.fit(X_train, Y_train)
# y_pred_xgb = xgb_classifier.predict(X_test)
#
# cm_xgb_classifier = confusion_matrix(Y_test, y_pred_xgb)
# print(cm_xgb_classifier)
#
# numerator = cm_xgb_classifier[0][0] + cm_xgb_classifier[1][1]
# denominator = sum(cm_xgb_classifier[0]) + sum(cm_xgb_classifier[1])
# acc_xgb = (numerator/denominator) * 100
# print('Accuracy : ', round(acc_xgb, 2), '%')
#
# cross_val_xgb = cross_val_score(estimator=XGBClassifier(), X=X_train, y=Y_train, cv=10, n_jobs=-1)
# print('Cross Validation Accuracy : ', round(cross_val_xgb.mean() * 100, 2), '%')
