import numpy as np
import copy
from collections import Counter
from nltk import word_tokenize
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
from sklearn.metrics import recall_score,f1_score,precision_score,accuracy_score, confusion_matrix, ConfusionMatrixDisplay 

def roundoff(number):
  if number >= 0.5:
    return 1
  else:
    return 0
  
def hyper_prameter_tune_mlp(X,y):
    mlp_gs = MLPClassifier(max_iter=3000)
    parameter_space = {
        'hidden_layer_sizes': [(200,),(400,),(600,)],
        'learning_rate_init':[0.1,0.01,0.001,0.0001],
        'learning_rate': ['constant','adaptive'],
    }
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
    clf.fit(X,y)

    print(clf.best_params_)

def checkAccuracy2(y_test, layer1, layer2):
  y_pred = []
  for i in range(len(layer1)):
    if(layer1[i] == 0):
      y_pred.append(layer1[i])
    else:
      y_pred.append(layer2[i])
  
  confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
  confusionMatrixDisplay.plot()
  plt.show()
  # check scores
  F1_score = f1_score(y_test, y_pred)
  Precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  print("Precision",Precision,"recall",recall,"F1_score",F1_score)
  print('Accuracy',accuracy_score(y_test, y_pred))

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
  


class confidence_classify:
  def __init__(self):
    dataset = "audio_dataset\\dataset.csv"
    file_paths = []
    for i in range(1,124):
      file_paths.append("D:\\machineLearningProject\\audio_dataset\\a" + str(i) + ".wav")
    df = pd.read_csv(dataset) # confidence and clarity of each recoring is stored in dataset.
    print(df.shape)
    mfccs = []
    
    max_length = 1500
    for f in file_paths:
      signal, sr = librosa.load(f)  #signal contains the amplitude of each sample in the audio with sampling rate of sr

      mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=11).T  

    # pad or truncate MFCC so that it can be of uniform size so that it can be placed into the ML model.
      if mfcc.shape[0] < max_length:
          padding = max_length - mfcc.shape[0]
          mfcc = np.pad(mfcc, ((padding,0), (0, 0)), mode='constant')
      else:
          mfcc = mfcc[-max_length - 1:-1, :]
      mfccs.append(mfcc)
      X_flattened = np.array([x.flatten() for x in mfccs])
      self.X = X_flattened
      self.y = df["confidence"].to_list()
    
    def modelComparison(self):
      X = self.X
      y = self.y
      knn = KNeighborsClassifier()
      mlp = MLPClassifier(max_iter=5000)
      dt = DecisionTreeClassifier()
      rf = RandomForestClassifier(n_estimators=944,max_features="sqrt",max_depth=70)
      print("knn",cross_val_score(knn,X,y,cv=5))
      print("rf",cross_val_score(rf,X,y,cv=5))
      print("mlp",cross_val_score(mlp,X,y,cv=5))
      print("dt",cross_val_score(dt,X,y,cv=5))

    def RandomForest_accuracy(self):  # to check accuracy of random forest model
      randomState = 60
      X_train, X_test, y_train, y_test = train_test_split(self.X,self.y, test_size=0.3, random_state=randomState)
      rf = RandomForestClassifier(n_estimators=944,max_features="sqrt",max_depth=70)
      rf.fit(X_train, y_train)
      checkAccuracy(rf,y_test,X_test)

    def HyperParameterTune(self):  # find optimal hyper-parameter tuning for random forest using random search
      n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
      # Number of features to consider at every split
      max_features = ['log2', 'sqrt',None]
      # Maximum number of levels in tree
      max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
      max_depth.append(None)

      random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    }
      clf = RandomForestClassifier()
      rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2,n_jobs=15)# Fit the random search model
      rf_random.fit(self.X, self.y)
      print(rf_random.best_params_)

    def stacking_model_test(self):  # test the working of stacking model
      y = self.y
      X = self.X
      randomState = 38
      clf1 = MLPClassifier(max_iter=4000, random_state=randomState)
      clf2 = RandomForestClassifier(random_state=randomState)
      clf3 = KNeighborsClassifier()

      X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=randomState)
      X_temp, X_test, y_temp, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=randomState)

      clf1.fit(X_train,y_train)
      clf2.fit(X_train,y_train)
      clf3.fit(X_train,y_train)
      m1 = clf1.predict(X_temp)
      m2 = clf2.predict(X_temp)
      m3 = clf3.predict(X_temp)

      meta_train = []
      for i in range(len(m1)):
        meta_train.append([m1[i],m2[i],m3[i]])

      clf_meta = AdaBoostClassifier(random_state=randomState)
      clf_meta.fit(meta_train,y_temp)

      ######################## CHECKING ACCURACY OF META MODEL ###########################
      m1_ = clf1.predict(X_test)
      m2_ = clf2.predict(X_test)
      m3_ = clf3.predict(X_test)
      meta_test = []
      for i in range(len(m1_)):
        meta_test.append([m1_[i],m2_[i],m3_[i]])
      y_pred = clf_meta.predict(meta_test)
      try:
        confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
        confusionMatrixDisplay.plot()
        plt.show()
      except ValueError:
        print(f"ERROR:------------------------------------------\n{len(y_pred)}\n{len(y_test)}")
      # check scores

      try:
        F1_score = f1_score(y_test, y_pred)
        Precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        print({"Precision":Precision,"recall":recall,"F1_score":F1_score})
        print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
      except ValueError:
        print("error......")

      y_pred_temp = clf1.predict(X_test)
      y_pred_layer1 = [roundoff(pr) for pr in y_pred_temp]
      print("prediction -> ",y_pred_layer1)


class clarity_classify:
  def __init__(self):   
    dataset = 'vivadata.csv'
    df = pd.read_csv(dataset,encoding='windows-1252')
    wc = Counter()
    for answer in df["answer"]:
      answer = str(answer)
      answer = answer.lower()
      wc.update(Counter(word_tokenize(answer)))

    max_stutter_order = 3
    stutters = []
    for answer in df['answer']:
      answer_words = word_tokenize(str(answer).lower())
      # the 3 commonly identified stutters
      isdbst = 0  # variable to check whether it is a double word stutter.
      istpst = 0  # to check whether it is a triple word stutter.

      double_word_stutters = 0
      triple_word_stutters = 0

      stutter = [0 for i in range(max_stutter_order)]

      for i in range(len(answer_words) - 1):
        isdbst = 0
        for j in range(max_stutter_order):
          if i < len(answer_words) - j - 1 and answer_words[i] == answer_words[i + j + 1]:
            stutter[j] += 1
            isdbst += 1
            istpst += 1
          else:
            isdbst -= 1
            istpst -= 1
          
        if isdbst == 2:
          double_word_stutters += 1
        if istpst == 3:
          triple_word_stutters += 1
      
      stutter.append(double_word_stutters)
      stutter.append(triple_word_stutters)
      stutters.append(stutter)

    for answer in df["answer"]:
      answer = str(answer)
      answer = answer.lower()
      wc.update(Counter(word_tokenize(answer)))

    # we only consider the most common words out of all
    com_words = dict(wc.most_common(40))

    vocabulary_count = dict(wc)
    vocabulary = list(wc.keys())

    # now take each of these words one by one, in order to predict for clarity.
    # we use term frequency vectorizer for these 40 words
    # instead of doing vectorization, another approach would be to remove each of the words in order and see the new length of the answer
    # we classify clarity based on the ratio between the previous length of the answer and the new length.

    tf_vectors = []
    for answer in df['answer']:
      answer = str(answer)
      answer = answer.lower()
      term_freq = []
      for word in com_words.keys():
        term_freq.append(answer.count(word))
      tf_vectors.append(term_freq)
    
    self.tf_vectors = tf_vectors
    self.stutters = stutters
    self.y = df['clarity'].values.astype('b')

  def stacking_model(self): 
    randomState = 42
    tf = copy.deepcopy(self.tf_vectors)
    for i in range(len(self.tf_vectors)):
      tf[i].extend(self.stutters[i])

    X = tf
    y = self.y

    smote=SMOTE(sampling_strategy='minority') 
    X,y=smote.fit_resample(X,y)

    # print('random state ->', randomState)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=randomState)


    clf1 = MLPClassifier(max_iter=4000)
    clf2 = RandomForestClassifier()

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=randomState)
    X_temp, X_test, y_temp, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=randomState)

    clf1.fit(X_train,y_train)
    clf2.fit(X_train,y_train)
    m1 = clf1.predict(X_temp)
    m2 = clf2.predict(X_temp)

    meta_train = []
    for i in range(len(m1)):
      meta_train.append([m1[i],m2[i]])

    clf_meta = DecisionTreeClassifier()
    clf_meta.fit(meta_train,y_temp)

    ######################## CHECKING ACCURACY OF META MODEL ###########################
    m1_ = clf1.predict(X_test)
    m2_ = clf2.predict(X_test)
    meta_test = []
    for i in range(len(m1_)):
      meta_test.append([m1_[i],m2_[i]])
    y_pred = clf_meta.predict(meta_test)
    try:
      confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
      confusionMatrixDisplay.plot()
      plt.show()
    except ValueError:
      print(f"ERROR:------------------------------------------\n{len(y_pred)}\n{len(y_test)}")
    # check scores

    try:
      F1_score = f1_score(y_test, y_pred)
      Precision = precision_score(y_test, y_pred)
      recall = recall_score(y_test, y_pred)
      print({"Precision":Precision,"recall":recall,"F1_score":F1_score})
      print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    except ValueError:
      print("error......")

    y_pred_temp = clf1.predict(X_test)
    y_pred_layer1 = [roundoff(pr) for pr in y_pred_temp]
    print("prediction -> ",y_pred_layer1)

  def layered_approach_test(self):
    randomState = random.randint(0,100)
    print('random state ->', randomState)

    X = self.stutters
    y = self.y

    # SMOTE is not used as the interpolated data may not be same for stutter and common word approach
    # smote=SMOTE(sampling_strategy='minority') 
    # X,y=smote.fit_resample(X,y)


    X_train, X_test_1, y_train, y_test_1 = train_test_split(X, y, test_size=0.3, random_state=randomState)

    clf = MLPClassifier(max_iter=4000)

    clf.fit(X_train,y_train)
    # checkAccuracy(clf,y_test,X_test)

    layer_1_predictions = clf.predict(X_test_1)

    X = self.tf_vectors

    clf1 = MLPClassifier(max_iter=4000)
    clf2 = RandomForestClassifier()
    print(clf1.get_params())
    print(clf2.get_params())
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=randomState)
    X_temp, X_test, y_temp, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=randomState)
    clf1.fit(X_train,y_train)
    clf2.fit(X_train,y_train)
    m1 = clf1.predict(X_temp)
    m2 = clf2.predict(X_temp)
    meta_train = []
    for i in range(len(m1)):
      meta_train.append([m1[i],m2[i]])

    clf_meta = DecisionTreeClassifier()
    clf_meta.fit(meta_train,y_temp)

    X_train, X_test_2, y_train, y_test_2 = train_test_split(X, y, test_size=0.3, random_state=randomState)
    m1_ = clf1.predict(X_test_2)
    m2_ = clf2.predict(X_test_2)
    meta_test = []
    for i in range(len(m1_)):
      meta_test.append([m1_[i],m2_[i]])
    y_pred = clf_meta.predict(meta_test)
    try:
      confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix(y_test_2, y_pred))
      confusionMatrixDisplay.plot()
      plt.show()
    except ValueError:
      print(f"ERROR:------------------------------------------\n{len(y_pred)}\n{len(y_test_2)}")
    # check scores

    try:
      F1_score = f1_score(y_test_2, y_pred)
      Precision = precision_score(y_test_2, y_pred)
      recall = recall_score(y_test_2, y_pred)
      print({"Precision":Precision,"recall":recall,"F1_score":F1_score})
      print(f'Accuracy: {accuracy_score(y_test_2, y_pred)}')
    except ValueError:
      print("error......")

    y_pred_temp = clf1.predict(X_test_2)

    y_pred_layer1 = y_pred_temp
    print("prediction -> ",y_pred_layer1)
    ########################################################################################################-------------------

    layer_2_predictions = y_pred

    checkAccuracy2(y_test_2,layer_1_predictions,layer_2_predictions)

  def dimensionality_reduction_mlp(self): # backward dimensionality reduction is performed
    randomState = 42
    tf = copy.deepcopy(self.tf_vectors)
    for i in range(len(self.tf_vectors)):
      tf[i].extend(self.stutters[i])

    X = tf
    y = self.y

    smote=SMOTE(sampling_strategy='minority') 
    X,y=smote.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=randomState)

    clf = MLPClassifier(max_iter=3000,hidden_layer_sizes=(200,),learning_rate_init=0.01)
    clf.fit(X_train,y_train)
        
    y_pred = clf.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    max_acc = acc_score
    print(max_acc)

    ogtf = copy.deepcopy(X)
    print(f"ogtf contains {len(ogtf)} rows and {len(ogtf[0])} columns")

    acc_increasing = True
    # commencing backward selection for dimensionality reduction
    while(acc_increasing):
      popout = -1
      acc_increasing = False
      for i in range(len(ogtf[0])): # len(ogtf[0]) contains the number of columns
        tf_vectors = copy.deepcopy(ogtf)
        try:
          for j in range(len(tf_vectors)):
            #print(j,str(len(tf_vectors[0])))
            tf_vectors[j].pop(i)
        except IndexError:
          
          print("trying to pop out " + str(i) + "while size of vector is " + str(len(tf_vectors[0])))
          exit()
        # now, out of these, we use dimensionality reduction to choose the most useful words
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=randomState)
        clf = MLPClassifier(max_iter=3000)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        if(acc_score > max_acc):
          popout = i
          acc_increasing = True
          max_acc = acc_score

      if (popout > -1):
        # popout particular column from all the rows
        print(f"popping out {popout}")
        for i in range(len(ogtf)):
          ogtf[i].pop(popout)
          # com_words.pop(com_words.keys()[popout])
        clf = MLPClassifier(max_iter=3000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=randomState)
        clf.fit(X_train,y_train)
        checkAccuracy(clf,y_test,X_test)

    print(max_acc)
    print(f"ogtf contains {len(ogtf)} rows and {len(ogtf[0])} columns")

  def only_stutter(self):
    X = self.stutters
    y = self.y

    smote=SMOTE(sampling_strategy='minority') 
    X,y=smote.fit_resample(X,y)

    # this is the stutter approach for identifying the clarity of what is spoken by the student
    # we vectorize the transcribed text based on the repetition of words. If a worda repeats right after each other, it is called a 0th order stutter
    # if a word repeats leaving 1 word gap, it is called a first order stutter and so on, if a word leaves gap of n words, before repeating, it is called an nth order stutter.
    # The model predicts the clarity based on the number of each of these stutters
    # the hyperparameters in this method consists of the ML model used and order of stutter being considered.
    # multiple methods were compared to find out the best ML model for stutters uptil 3rd order. 
    # MLP classifier was chosen for the classification based on this method.
    # Then the next the order of words to be chosen is chosen by backward reduction techniques.
    # It was observed that uptil 3rd order of stutters probided best accuracy
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
  # X_temp, X_test, y_temp, y_test = train_test_split(X_temp, y_temp, test_size=0.99, random_state=randomState)

    clf = MLPClassifier(max_iter=3000)

    X_test = X_temp
    y_test = y_temp
    clf.fit(X_train,y_train)
    checkAccuracy(clf,y_test,X_test)


            