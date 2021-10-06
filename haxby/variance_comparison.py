
# TODO: get data for both masks
# TODO: work out how to do the variance comparison
# have the X and have the Y,
#
# should variance be done for each class separately or all fuctional data at once, find variance statistic fuction to see input format
# if so, will need to reorganise data for variance comparison
# load up the functional data and relevant masks
# standardise/normalise fmri data first(?)


# TODO: 1. Take only relevant voxels from mask.(come back to this step) Transform the data by subtracting the mean (or median) from each data point and take the absolute values.
#  Now check the normality of each sample again and use a t-test or KS test as appropriate.
#whole_group_data =
# remove rest data
import pandas as pd
from nilearn.input_data import NiftiMasker
from scipy.stats import ansari
import copy
import numpy as np
from iteration_utilities import flatten



subjects = [1,2,3,4,5]
# for now had to reduce all to 11 examples,
# perhaps should be taking average activation for each category for variance comparison
def convert_lst_to_array(lst):
    arr = None
    result = []
    for sublst in lst:
        arr = np.array(sublst[0])
        for item in sublst[1:]:
            arr = np.vstack((arr,np.array(item)))
        result.append(arr)
    return np.array(result)

def organise_data(data, label, subject,alldic):
    dic = {'scissors': [], 'chair': [], 'face': [],
           'shoe': [], 'cat': [], 'bottle': [],
           'scrambledpix': [], 'house': []}
    label = label.tolist()
    for timestep in range(0, len(data), 9):
        category = label[timestep]
        print("The category is ", category)
        relevantdata = data[timestep:timestep+9]
        dic[category].append(relevantdata)
    for key in dic.keys():
        dic[key] = dic[key][0:11]
    print("How many examples per category ", len(dic['scissors']),len(dic['bottle']))
    alldic[subject] = dic
   # print(alldic.keys())

def process_labels(subject):
    label_filename = 'C:/Users/hlw69/nilearn_data/haxby2001/subj' + str(subject) + '/labels.txt'
    labels = pd.read_csv(label_filename, sep=" ")
    y = labels['labels']
    # Remove the rest condition, it is not very interesting
    non_rest = (y != 'rest')
    y = y[non_rest]
  #  print(y)
    # Remove the "rest" condition
    session = labels['chunks']
    return session, y, non_rest

# load up data
def load_data(subject, mask):
    print("working with subject ", subject)
    mask_filename = 'C:/Users/hlw69/nilearn_data/haxby2001/masks/' + mask + '.nii'
    func_filename = 'C:/Users/hlw69/nilearn_data/haxby2001/subj' + str(subject) + '/' + '4D.nii'
    session, y, non_rest = process_labels(subject)
    # For decoding, standardizing is often very important
    nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True,
                               sessions=session, smoothing_fwhm=4,
                               memory="nilearn_cache", memory_level=1)
    X = nifti_masker.fit_transform(func_filename)
    X = X[non_rest]
    session = session[non_rest]

    return X, y

def calculate_variance(x1,x2):
    print("calculating variance")
    stat, p = wilcoxon(x1, y=x2, alternative = 'two-sided')
    print("Wilxocon results are ", stat, p )
    statistic, p = ansari(x1, x2)
    print("Ansari results are ", statistic, p)
    return statistic,p

def flatten_dictionary_data(subject, alldic):
    flattened = []
    for key in alldic[subject].keys():
        flattened.append(alldic[subject][key])
    flat = list(flatten(flattened))
    flat = list(flatten(flat))
    #print(len(flat)) # 8, 12, 9 , 25096-> 864, 25096
    return flat

# save numpy array as csv file
from numpy import asarray

# save numpy array as npy file
from numpy import asarray
from numpy import save


# save numpy array as npy file
from numpy import asarray
# define data
# save to csv file
masks = ['maskbounded17_18_19_dilated_1','maskbounded20_37_d1']
def compute():
    samples = []
    for mask in masks:
        subjects_flattened_data = []
        alldic = {}
        for subject in subjects:
            X,y = load_data(subject, mask)
            organise_data(X,y,subject,alldic)
            subjects_flattened_data.append(flatten_dictionary_data(subject,alldic))
        x = convert_lst_to_array(subjects_flattened_data)
        save(mask + 'data.npy', x)
        print(x.shape)
        samples.append(x)
    return x

from matplotlib import pyplot


x = compute()

from scipy.stats import shapiro, wilcoxon


# seed the random number generator
# normality test
stat, p = shapiro(x[0])
print("Is the data normal? high p indicates normal  for x0", stat, p)
stat, p = shapiro(x[1])
print("Is the data normal? high p indicates normal  for x1", stat, p)
calculate_variance(x[0], x[1])

pyplot.hist(x[1])
pyplot.show()

#print("before conversion ", len(subjects_flattened_data), len(subjects_flattened_data[0]), len(subjects_flattened_data[0][0]))


#x1 = convert_lst_to_array(subjects_flattened_data)
#print("after array conversion ", x1 .shape)
#x2 = copy.deepcopy(x1) + np.random.poisson(2, x1.shape)




  #  for timestep in range(data):
  #      data[timestep] = data[timestep] - data[timestep].mean()



    #  Now check the normality of each sample again and use a t-test or KS test as appropriate.







