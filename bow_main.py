#!/usr/bin/env python



#Importing the required libraries
import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import autosklearn.classification
from sklearn.model_selection import GridSearchCV

#Preparing the dataset 
path = '../vision_jw_dataset'
name_list = ['ALU_p0', 'ALU_p1', 'ALU_p2', 'ALU_p3', 'ALU_p4', 'ALU_p5', 'ALU_p6']
image_path = []
for i in range(7):
  dir = os.path.join(path, name_list[i])
  for file in os.listdir(dir):
    image_path.append(os.path.join(dir, file))


mode = 'sift'


def main(thresh, k = 32):

  t0 = time.time()


  def CalcFeatures(img, th):
    if mode == 'sift':
      sift = cv2.SIFT_create()
      kp, des = sift.detectAndCompute(img, None)
    elif mode == 'orb':
      orb = cv2.ORB_create()
      kp = orb.detect(img,None)
      kp, des = orb.compute(img, kp)
    elif mode == 'brief':
      star = cv2.xfeatures2d.StarDetector_create()
      brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
      kp = star.detect(img,None)
      kp, des = brief.compute(img, kp)
    elif mode == 'surf':
      surf = cv2.xfeatures2d.SURF_create(th)
      kp, des = surf.detectAndCompute(img,None)

    # print(des.shape)
    des = np.asarray(des[:th, :], dtype=np.float32)

    return des
  
  '''
  All the files appended to the image_path list are passed through the
  CalcFeatures functions which returns the descriptors which are 
  appended to the features list and then stacked vertically in the form
  of a numpy array.
  '''

  features = []
  for file in image_path:
    img = cv2.imread(file, 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = img[300:900, 400:1200]
    img_des = CalcFeatures(img, thresh)
    if img_des is not None:
      features.append(img_des)
  features = np.vstack(features)
  print(features.shape)

  '''
  K-Means clustering is then performed on the feature array obtained 
  from the previous step. The centres obtained after clustering are 
  further used for bagging of features.
  '''

  # k = 32
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
  flags = cv2.KMEANS_RANDOM_CENTERS
  compactness, labels, centres = cv2.kmeans(features, k, None, criteria, 10, flags)

  '''
  The bag_of_features function assigns the features which are similar
  to a specific cluster centre thus forming a Bag of Words approach.  
  '''

  def bag_of_features(features, centres, k = 500):
      vec = np.zeros((1, k))
      for i in range(features.shape[0]):
          feat = features[i]
          diff = np.tile(feat, (k, 1)) - centres
          dist = pow(((pow(diff, 2)).sum(axis = 1)), 0.5)
          idx_dist = dist.argsort()
          idx = idx_dist[0]
          vec[0][idx] += 1
      return vec

  labels = []
  vec = []
  for file in image_path:

    img = cv2.imread(file, 0)
    # img = img[300:900, 400:1200]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_des = CalcFeatures(img, thresh)
    if img_des is not None:
      img_vec = bag_of_features(img_des, centres, k)
      vec.append(img_vec)
      labels.append(int(file[26]))
  vec = np.vstack(vec)

  '''
  Splitting the data formed into test and split data and training the 
  SVM Classifier.
  '''

  X_train, X_test, y_train, y_test = train_test_split(vec, labels, test_size=0.1)

  ## SVM
  if False:
    params = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100, 1000, 10000], 'gamma': [1,0.1,0.01,0.001, 0.0001, 0.00001]}
    svc = SVC()
    clf = GridSearchCV(svc, params)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    conf_mat = confusion_matrix(y_test, preds)
  
  else:
    ## automl
    clf = autosklearn.classification.AutoSklearnClassifier()
    clf.fit(X_train, y_train) 
    print(clf.sprint_statistics())
    # print(clf.leaderboard())
    try:
      preds = clf.predict(X_test).argmax(axis=1)
      acc = accuracy_score(y_test, preds)
      conf_mat = confusion_matrix(y_test, preds)
    except:
      acc=0
      conf_mat = []
      # print(preds)
  
  t1 = time.time()
  
  return acc*100, conf_mat, (t1-t0)


accuracy = []
timer = []

param_list=[]
# for i in range(1):
  # for j in range(6):
    # param_list.append([2**(i+16), 2**(j+4)])

param_list.append([40960000,128])
# param_list.append([40960000,128])
# param_list.append([40960000,128])
# param_list.append([40960000,128])
# param_list.append([40960000,128])


print('model : brief')


mode = 'brief'

for param in param_list:
  feat_num = param[0]
  k = param[1]
  print('param : {}, {}'.format(param[0], param[1]))
  data = main(feat_num, k = k)
  accuracy.append(data[0])
  conf_mat = data[1]
  timer.append(data[2])
  print('Accuracy = {}\nTime taken = {} sec\nConfusion matrix :\n{}'.format(data[0],data[2],data[1]))


# print('model : sift')


# param_list=[]
# # for i in range(4):
#   # for j in range(6):
#     # param_list.append([2**(i+10), 2**(j+4)])
# # param_list.append([4096, 64])
# # param_list.append([4096, 128])
# # param_list.append([4096*2, 64])
# # param_list.append([4096*2, 128])
# # param_list.append([4096*4, 64])
# param_list.append([4096*40, 128])
# param_list.append([4096*40, 128])
# param_list.append([4096*40, 128])
# param_list.append([4096*40, 128])
# param_list.append([4096*40, 128])

# mode = 'sift'

# for param in param_list:
#   feat_num = param[0]
#   k = param[1]
#   print('param : {}, {}'.format(param[0], param[1]))
#   data = main(feat_num, k = k)
#   accuracy.append(data[0])
#   conf_mat = data[1]
#   timer.append(data[2])
#   print('Accuracy = {}\nTime taken = {} sec\nConfusion matrix :\n{}'.format(data[0],data[2],data[1]))


