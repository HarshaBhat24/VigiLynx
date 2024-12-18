import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix,classification_report
import joblib
from sklearn import cross_decomposition
data=pd.read_csv('https://raw.githubusercontent.com/mburakergenc/Malware-Detection-using-Machine-Learning/refs/heads/master/data.csv ')
data_in=data.drop(['hash','classification'], axis=1).values
labels=data['classification'].values
extratree=ExtraTreesClassifier().fit(data_in,labels)
select=SelectFromModel(extratree,prefit=True)
data_in_new=select.transform(data_in)
legit_train,legit_test,mal_train,mal_test=train_test_split(data_in_new,labels,test_size=0.2)
classif=RandomForestClassifier(n_estimators=50)
classif.fit(legit_train,mal_train) 
accuracy=classif.score(legit_test,mal_test)
result=classif.predict(legit_test)
conf_mat=confusion_matrix(mal_test,result)
print(conf_mat)
print("Classificaion report")
print(classification_report(mal_test,result))
print("Accuracy: ",accuracy)
joblib.dump(classif,'model.pkl')
new_sample = [[0, 0, 0, 3069378560, 14274, 0, 0, 0, 13173, 0, 0, 0, 724, 6850, 0, 150, 120, 124, 210, 0, 120, 3473, 341974, 0, 0, 120, 0, 3204448256, 380690, 4, 0, 0, 0]]
new_sample_transformed = select.transform(new_sample)
prediction = classif.predict(new_sample_transformed)

# Interpret the prediction
if prediction[0] == 'malware':
    print("The software is classified as MALWARE.")
else:
    print("The software is classified as BENIGN.")








