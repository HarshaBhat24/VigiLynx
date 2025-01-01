import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

file = "phishing.csv"
data = pd.read_csv(file)

data['status'] = data['status'].map({'phishing': 1, 'legitimate': 0})

x = data.drop(['status'], axis=1)
y = data['status']

x = x.dropna(axis=1, how='all')

numeric_cols = x.select_dtypes(include=['number']).columns
categorical_cols = x.select_dtypes(include=['object']).columns

imputer = SimpleImputer(strategy='mean')
x_numeric = pd.DataFrame(imputer.fit_transform(x[numeric_cols]), columns=numeric_cols)

x_categorical = pd.get_dummies(x[categorical_cols], drop_first=True)

x_processed = pd.concat([x_numeric, x_categorical], axis=1)

assert x_processed.shape[1] == x_numeric.shape[1] + x_categorical.shape[1], "Column mismatch detected!"

x_train, x_test, y_train, y_test = train_test_split(x_processed, y, test_size=0.2, random_state=42)


models={
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
}
def evaluate_model(model, x_train, y_train, x_test, y_test):
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores=cross_val_score(model, x_train, y_train, cv=cv, scoring='accuracy')
    print(f"{type(model).__name__} Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
def detailed_eval(model,x_train,y_train,x_test,y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"{type(model).__name__} Accuracy :")
    print(classification_report(y_test, y_pred))
    print(f"{type(model).__name__} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(x_test)[:,1]


        fpr,tpr,threshold=roc_curve(y_test, y_pred_proba)
        roc_auc=auc(fpr,tpr)
        plt.figure(figsize=(8,6))
        plt.plot(fpr,tpr,label=f'{type(model).__name__} (area={roc_auc:.2f})')
        plt.plot([0,1],[0,1],color='navy',linestyle='--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {type(model).__name__}')
        plt.legend(loc="lower right")
        plt.show()
    else:
        print(f"{type(model).__name__} does not support predict_proba")
        print("ROC curve and AUC score not available")


for model_name, model in models.items():
    evaluate_model(model, x_train, y_train, x_test, y_test)
    detailed_eval(model, x_train, y_train, x_test, y_test)

    





