import numpy as np
import cv2
import glob, os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from skimage.feature import graycomatrix, graycoprops
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def get_classes(dataset_path):
    idx_2_class, class_2_idx = {}, {}
    folders = sorted(glob.glob(dataset_path + "/*"))
    for i, f in enumerate(folders):
        c = os.path.basename(f)
        c = c[c.find('p'):]
        if not c in class_2_idx:
            idx_2_class[i] = c
            class_2_idx[c] = i
    return idx_2_class, class_2_idx

def get_label(path):
    label = os.path.basename(os.path.dirname(path))
    return label[label.find('p'):]
        

def calc_glcm(img):
    glcm = graycomatrix(img, distances=[5, 10, 15, 20, 25], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], levels=256,
                        symmetric=True, normed=True)

    glcm_props = []
    glcm_props.extend(graycoprops(glcm, 'energy'))
    glcm_props.extend(graycoprops(glcm, 'dissimilarity'))
    glcm_props.extend(graycoprops(glcm, 'correlation'))
    glcm_props = np.array(glcm_props)
    glcm_props = glcm_props.flatten()
    
    return glcm_props
    

def get_data(class_2_idx, opt='train'):
    if not os.path.exists('output/glcm/preprocessed_{}_data.pkl'.format(opt)):
        X, y = [], []
        dataset = sorted(glob.glob("vision_dataset/{}/*/*".format(opt)))
        for i, data in enumerate(dataset):
            label = get_label(data)
            img = cv2.imread(data)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            glcm_props = calc_glcm(gray_img)
            X.append(glcm_props)
            y.append(class_2_idx[label])
            print('Processing data:', i)
        with open('output/glcm/preprocessed_{}_data.pkl'.format(opt),"wb") as f:
                pickle.dump(X, f)
                pickle.dump(y, f)
    
    else:
        with open('output/glcm/preprocessed_{}_data.pkl'.format(opt), "rb") as f:
            X = pickle.load(f)
            y = pickle.load(f)
    
    return X, y

def train_knn(X_train, y_train, k_range=[3, 11], cv=10):
    # list of scores from k_range
    param_grid = dict(n_neighbors=list(range(k_range[0], k_range[1])))

    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', return_train_score=False,verbose=1)

    grid.fit(X_train, y_train)

    print(grid.best_params_)
    grid_mean_scores = [result for result in grid.cv_results_['mean_test_score']]
    plt.plot(range(k_range[0], k_range[1]), grid_mean_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')

    # Save best parameter network
    knnPickle = open('output/glcm/knnpickle_file.pkl', 'wb') 
    # source, destination 
    pickle.dump(grid.best_estimator_, knnPickle)  
    # close the file
    knnPickle.close()
    plt.show()


def train_svm(X_train, y_train, cv=10):
    # list of scores from k_range
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 1, cv=cv)   
    # fitting the model for grid search
    grid.fit(X_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)
    
    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    # Save best parameter network
    svmPickle = open('output/glcm/svmpickle_file.pkl', 'wb') 
    # source, destination 
    pickle.dump(grid.best_estimator_, svmPickle)  
    # close the file
    svmPickle.close()


def main(opt, method):
    if not os.path.exists('output/glcm'):
        os.makedirs('output/glcm')

    idx_2_class, class_2_idx = get_classes("vision_dataset/train")

    if opt == 'train':
        X_train, y_train = get_data(class_2_idx, 'train')
        if method == 'knn':
            train_knn(X_train, y_train, [1, 9], 10)
        elif method == 'svm':
            train_svm(X_train, y_train)
        else:
            print('Such methods does not exist')


    else:
        # load the model from disk
        loaded_model = pickle.load(open('output/glcm/{}pickle_file.pkl'.format(method), 'rb'))

        X_test, y_test = get_data(class_2_idx, 'test')

        result = loaded_model.predict(X_test) 

        # Accuracy
        print('Accuracy:', metrics.accuracy_score(y_test, result))

        # Confusion Matrix
        confusion_matrix = metrics.confusion_matrix(y_test, result)
        ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
        ax.set_title('Seaborn Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(class_2_idx.keys())
        ax.yaxis.set_ticklabels(class_2_idx.keys())

        ## Display the visualization of the Confusion Matrix.
        plt.show()


if __name__ == "__main__":
    main('train', 'knn')