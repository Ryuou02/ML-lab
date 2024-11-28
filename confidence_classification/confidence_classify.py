import numpy as np
#from sklearn.cluster import KMeans
import random
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score,f1_score,precision_score,accuracy_score, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB  
import matplotlib.pyplot as plt
import seaborn as sn
import random
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV,cross_validate
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics
import librosa
from sklearn.neural_network import MLPClassifier
import pandas as pd
#from playsound import playsound
from sklearn.metrics import recall_score,f1_score,precision_score,accuracy_score, confusion_matrix, ConfusionMatrixDisplay 

def checkAccuracy(clf, y_test, X_test):
  y_pred = clf.predict(X_test)
  try:
    confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    confusionMatrixDisplay.plot()
    plt.show()
  except ValueError:
    print(f"ERROR:------------------------------------------\n{y_pred}\n{y_test}")
  # check scores

  try:
    F1_score = f1_score(y_test, y_pred)
    Precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print({"Precision":Precision,"recall":recall,"F1_score":F1_score})
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
  except ValueError:
    print("error......")
  


file_paths = []
for i in range(1,124):
  file_paths.append("D:\\machineLearningProject\\audio_dataset\\a" + str(i) + ".wav")

df = pd.read_csv("audio_dataset\\dataset.csv") # confidence and clarity of each recoring is stored in dataset.
print(df.shape)
mfccs = []

max_length = 1500
for f in file_paths:
  signal, sr = librosa.load(f)  #signal contains the amplitude of each sample in the audio with sampling rate of sr

  mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=3).T  

# pad or truncate MFCC so that it can be of uniform size so that it can be placed into the ML model.
  if mfcc.shape[0] < max_length:
      # librosa.display.specshow(mfcc, y_axis='time')
      # plt.show()
      padding = max_length - mfcc.shape[0]
      mfcc = np.pad(mfcc, ((padding,0), (0, 0)), mode='constant')
      # librosa.display.specshow(mfcc, y_axis='time')
      # plt.show()
  else:
      # librosa.display.specshow(mfcc, y_axis='time')
      # plt.show()
      mfcc = mfcc[-max_length - 1:-1, :]
      # librosa.display.specshow(mfcc, y_axis='time')
      # plt.show()
  mfccs.append(mfcc)

X_flattened = np.array([x.flatten() for x in mfccs])
X = X_flattened
y = df["confidence"].to_list()



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=random.randint(0,100))

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(max_iter=1000)
knn = KNeighborsClassifier()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()



print("knn",cross_val_score(knn,X,y,cv=5))
print("rf",cross_val_score(rf,X,y,cv=5))
print("mlp",cross_val_score(mlp,X,y,cv=5))
print("dt",cross_val_score(dt,X,y,cv=5))


knn.fit(X_train, y_train)
# knn.fit(X_test,y_test)

# checkAccuracy(knn,y_test,X_test)
