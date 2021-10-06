from nilearn import datasets
import numpy as np
import pandas as pd
# By default 2nd subject from haxby datasets will be fetched.
haxby_dataset = datasets.fetch_haxby(subjects=4)

# Print basic information on the dataset
print('Mask nifti images are located at: %s' % haxby_dataset.mask)
print('Functional nifti images are located at: %s' % haxby_dataset.func[0])

func_filename = haxby_dataset.func[0]
mask_filename = haxby_dataset.mask

# Load the behavioral data that we will predict
labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
y = labels['labels']
session = labels['chunks']

# Remove the rest condition, it is not very interesting
non_rest = (y != 'rest')
y = y[non_rest]

# Get the labels of the numerical conditions represented by the vector y
unique_conditions, order = np.unique(y, return_index=True)
# Sort the conditions by the order of appearance
unique_conditions = unique_conditions[np.argsort(order)]


from nilearn.input_data import NiftiMasker
# For decoding, standardizing is often very important
nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True,
                           sessions=session, smoothing_fwhm=4,
                           memory="nilearn_cache", memory_level=1)
X = nifti_masker.fit_transform(func_filename)

# Remove the "rest" condition
X = X[non_rest]
session = session[non_rest]

from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline

svc_ovo = OneVsOneClassifier(Pipeline([
    ('anova', SelectKBest(f_classif, k=500)),
    ('svc', SVC(kernel='linear'))
]))

svc_ova = OneVsRestClassifier(Pipeline([
    ('anova', SelectKBest(f_classif, k=500)),
    ('svc', SVC(kernel='linear'))
]))


from sklearn.model_selection import cross_val_score

cv_scores_ovo = cross_val_score(svc_ovo, X, y, cv=5, verbose=1)

cv_scores_ova = cross_val_score(svc_ova, X, y, cv=5, verbose=1)

print('OvO:', cv_scores_ovo.mean())
print('OvA:', cv_scores_ova.mean())

from matplotlib import pyplot as plt
plt.figure(figsize=(4, 3))
plt.boxplot([cv_scores_ova, cv_scores_ovo])
plt.xticks([1, 2], ['One vs All', 'One vs One'])
plt.title('Prediction: accuracy score')
plt.show()