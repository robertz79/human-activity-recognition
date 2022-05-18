import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV ,RepeatedKFold,cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,precision_score,recall_score,f1_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,f_classif, mutual_info_classif,RFE,chi2
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import RepeatedStratifiedKFold
import pymrmr
from sklearn.calibration import CalibratedClassifierCV
from mrmr import mrmr_classif
from imblearn.over_sampling import SMOTE,SVMSMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle
import seaborn as sns   
import timeit
from flask import Flask,request,jsonify
class specializedـknn_feature4:
    def __init__(self,selector=None):
        self.Pipe=None
        self.selector=selector
    def fit(self,x_train,y_train):
        self.Pipe=[]
        Y_train=[]
        for i in range(1,35):
            Y_train.append(np.zeros(len(x_train)))
            for j in range(len(x_train)):
                if y_train[j]==i:
                    Y_train[i-1][j]=1            
        for i in range(34):
                pipe= Pipeline([('classifier',KNeighborsClassifier(n_neighbors=2,metric='euclidean',weights='distance'))])
                pipe.fit(x_train[:,self.selector[i]],Y_train[i])
                self.Pipe.append(pipe)                    
        return self
        
    def predict(self,x_test):   
        d=[]
        yy_pred=[]
        c=[]
        for i in range(34):
            c.append(self.Pipe[i].predict_proba(x_test[:,self.selector[i]]))              
        for j in range(len(x_test)): 
            for i in range(34):
                d.append(c[i][j,1])  
            yy_pred.append(d.index(max(d))+1)
            d=[]   
        return yy_pred   
filename = 'model-non-sp.sav'
model1= pickle.load(open(filename, 'rb'))
filename = 'model-with-sp.sav'
model2= pickle.load(open(filename, 'rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict1',methods=['POST'])
def predict1():
    a = request.form.get('cgpa')
    b1= np.zeros(744).reshape(1,-1)
    for i in range(len(b1)):
        b1[i]=a  
    result = model1.predict(b1)
    return jsonify({'placement':str(result)})
if __name__ == '__main__':
    app.run(debug=True)





