from sklearn.model_selection import train_test_split
import glob, os, shutil

# Load Original dataset
files = sorted(glob.glob('vision_project/*/*.jpg'))
X, y = [], []
for f in files:
    X.append(f)
    label = os.path.basename(os.path.dirname(f))
    y.append(label)

# Make split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
print(len(X_train))
print(len(X_test))

# Create dataset path
if not os.path.exists('vision_dataset/train'):
    os.makedirs('vision_dataset/train')
if not os.path.exists('vision_dataset/test'):
    os.makedirs('vision_dataset/test')

# Train files
for xt, yt in zip(X_train, y_train):
    folder = 'vision_dataset/train/{}'.format(yt)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    f_name = folder + '/' + os.path.basename(xt)
    shutil.copy(xt, f_name)

# Test files
for xt, yt in zip(X_test, y_test):
    folder = 'vision_dataset/test/{}'.format(yt)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    f_name = folder + '/' + os.path.basename(xt)
    shutil.copy(xt, f_name)