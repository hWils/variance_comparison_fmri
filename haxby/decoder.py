#https://nilearn.github.io/auto_examples/plot_decoding_tutorial.html#retrieve-and-load-the-fmri-data-from-the-haxby-study
#https://nilearn.github.io/auto_examples/02_decoding/plot_haxby_full_analysis.html
# load up data

"""
    4D fMRI timeseries image. (1452 volumes with 40 x 64 x 64 voxels,
    corresponding to a voxel size of 3.5 x 3.75 x 3.75 mm and a volume repetition
    time of 2.5 seconds). The timeseries contains all 12 runs of the original
    experiment, concatenated in a single file. Please note, that the timeseries
    signal is *not* detrended.

"""



# TODO: Print out the scores for each separate class.
# TODO: Could make into a dictionary format
# create new masks (V1, V2, V3, V4)    (ITP- check brodmans)


import nilearn
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline
from nilearn.image import index_img
# Load nilearn NiftiMasker, the practical masking and unmasking tool
from nilearn.input_data import NiftiMasker
import numpy as np
from nilearn.decoding import Decoder
# Make a data splitting object for cross validation
from sklearn.model_selection import LeaveOneGroupOut
# Fetch data using nilearn dataset fetcher
from nilearn import datasets
# by default we fetch 2nd subject data for analysis
#haxby_dataset = datasets.fetch_haxby(subjects=1)
#func_filename = haxby_dataset.func[1] # ‘func’: string list. Paths to nifti file with bold data.
subjects = [1,2,3,4,5] #, 2, 3, 4, 5, 6]
""" 
Loads up the functional data for each subjct and stores as a list. 
func_anat can be 'bold' to get functional, or 'anat' to get anatomical data
"""
def load_niigz(func_anat = '4D'):
    all_func = []
    for sub in subjects:
        if func_anat == 'anat' and sub ==6:
            print("subject 6 anat missing so skipping")
        else:
            filename = 'C:/Users/hlw69/nilearn_data/haxby2001/subj' + str(sub) +'/'+func_anat +'.nii'
            data = nilearn.image.load_img(filename)
            print(filename)
            print("shape of data is ", data.shape, " for ", func_anat)
            all_func.append(data)
    return all_func

"""
Loads 
"""
def load_labels():
    all_labels = []
    for subject in subjects:
        print(subject)
        filename = 'C:/Users/hlw69/nilearn_data/haxby2001/subj' + str(subject) +'/labels.txt'
        label = pd.read_csv(filename, sep=" ")
        print(label)
        all_labels.append(label)
    return all_labels
"""
# get everything loaded
functional_data = load_niigz()
print("functional data is loaded")
anatomical_data = load_niigz(func_anat='anat')
print("anatomical data is loaded")
labels = load_labels()
print("labels are loaded")

# segment data
all_subjects_stimuli = []
task_masks = []
all_subjects_taskdata = []
categories = []
task_data = []
session_labels = []
"""
def data_segmentation(subject):
    subject -=1
   # print(labels[subject]['labels'])
    stimuli = labels[subject]['labels']
    all_subjects_stimuli.append(stimuli)
    task_mask = (stimuli != 'rest')
    print("task masks : ", subject, task_mask)
    task_masks.append(task_mask)
    categories.append(stimuli[task_mask].unique()) #get the names of the different cateogires
    print("session label? ", labels[subject]["chunks"][task_mask])
    session_labels.append(labels[subject]["chunks"][task_mask]) # hmmmm task mask are booleans, so session labels removes rest maybe?
    print(functional_data[subject].shape, task_mask.shape)
    print("index imge produces ", index_img(functional_data[subject], task_mask).shape)
    task_data.append(index_img(functional_data[subject], task_mask)) #extracts only the functional data at timepoints relevant to the tasks, gets rid of the 'rest' data
    all_subjects_taskdata.append(task_data)



def improved_data_segmentation(subject = 1, mask= 'maskbounded17_18_19_dilated_1', func_anat = '4D'):
    mask_filename = 'C:/Users/hlw69/nilearn_data/haxby2001/masks/' + mask + '.nii'
    label_filename = 'C:/Users/hlw69/nilearn_data/haxby2001/subj' + str(subject) + '/labels.txt'
    func_filename = 'C:/Users/hlw69/nilearn_data/haxby2001/subj' + str(subject) + '/' + func_anat + '.nii'
    labels = pd.read_csv(label_filename, sep=" ")
    #labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ") #'session_target': string list. Paths to text file containing session and target data.
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
    return X, y, session, unique_conditions



svc_ovo = OneVsOneClassifier(Pipeline([
    #('anova', SelectKBest(f_classif, k=500)),
    ('svc', SVC(kernel='linear'))
]))

svc_ova = OneVsRestClassifier(Pipeline([
    #('anova', SelectKBest(f_classif, k=500)),
    ('svc', SVC(kernel='linear'))
]))

mask_names = ['maskbounded20_37_d1']#['maskbounded17_18_19_dilated_1']
for mask in mask_names:
    print("With ", mask, ' applied:')
    for subject in subjects:

        print(subject)
        X, y, session, unique_conditions = improved_data_segmentation(subject,mask)
       # print("y the labels are ", y.shape, y)
       # print("X the functional data is ", X.shape, X)
        cv_scores_ovo = cross_val_score(svc_ovo, X, y, cv=5, verbose=0)
        cv_scores_ova = cross_val_score(svc_ova, X, y, cv=5, verbose=0)
        print('For subject ', subject, ' OvO:', cv_scores_ovo.mean())
        print('For subject ', subject, 'OvA:', cv_scores_ova.mean())

        from matplotlib import pyplot as plt
        plt.figure(figsize=(4, 3))
        plt.boxplot([cv_scores_ova, cv_scores_ovo])
        plt.xticks([1, 2], ['One vs All', 'One vs One'])
        plt.title('Prediction: accuracy score')
       # plt.show()
   # data_segmentation(subject)

# session labels are used as ways to split data into testing and training

"""
Load up early visual cortex and later area masks. These should be .nii files
"""

#early_mask = nilearn.image.load_img(filename)
#late_mask = nilearn.image.load_img(filename)

# mask outside of subject files full brain(?) - full brain is where all voxels unrelated to brain are removed.
# load up mask and see how it looks.
#print(stimuli)

# Print basic information on the dataset
#print('First subject anatomical nifti image (3D) located is at: %s' %
#      haxby_dataset.anat[0])  # anat’: string list. Paths to anatomic images.
#print('First subject functional nifti image (4D) is located at: %s' %
#      func_filename)

# for the original decoding pipeline
#cv = LeaveOneGroupOut()
#mask_names = ['maskbounded20_37_d1']#['maskbounded17_18_19_dilated_1']   #['rB1']  #, 'mask8_face_vt', 'mask8_house_vt']
mask_scores = {}
mask_chance_scores = {}

def original_decoding(subject):
    print("For subject %s", subject)
    subject -=1
    for mask_name in mask_names:
        print("Working on %s" % mask_name)
        # For decoding, standardizing is often very important
        haxby_mask_filename =  'C:/Users/hlw69/nilearn_data/haxby2001/subj' + str(subject+1) +'/'+mask_name +'.nii.gz'#  str(subject+1)
        mask_filename = 'C:/Users/hlw69/nilearn_data/haxby2001/masks/' + mask_name + '.nii'
        masker = NiftiMasker(mask_img=mask_filename, standardize=True)
        mask_scores[mask_name] = {}
        mask_chance_scores[mask_name] = {}
        print(categories[subject])
        for category in categories[subject]:
            print("Processing %s %s" % (mask_name, category))
            classification_target = (all_subjects_stimuli[subject][task_masks[subject]] == category)
            print("classification target is ", classification_target)
            # Specify the classifier to the decoder object.
            # With the decoder we can input the masker directly.
            # We are using the svc_l1 here because it is intra subject.
            decoder = Decoder(estimator='svc_l1', cv=cv,
                              mask=masker, scoring='roc_auc')
            decoder.fit(task_data[subject], classification_target, groups=session_labels[subject])
            mask_scores[mask_name][category] = decoder.cv_scores_[1]
            print("Scores: %1.2f +- %1.2f" % (
                  np.mean(mask_scores[mask_name][category]),
                  np.std(mask_scores[mask_name][category])))

            dummy_classifier = Decoder(estimator='dummy_classifier', cv=cv,
                                       mask=masker, scoring='roc_auc')
            dummy_classifier.fit(task_data[subject], classification_target,
                                 groups=session_labels[subject])
            mask_chance_scores[mask_name][category] = dummy_classifier.cv_scores_[1]


#for subject in subjects:
 #   original_decoding(subject)
