import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import cross_decomposition
data=pd.read_csv('https://raw.githubusercontent.com/mburakergenc/Malware-Detection-using-Machine-Learning/refs/heads/master/data.csv ')
legit_train,legit_test,mal_train,mal_test=cross_decomposition.train_test_split(data.drop(['hash','classification'], axis=1).values,data['classification'].values,test_size=0.2)