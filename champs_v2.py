'''
Author: Rueen Fiez & Parth Patel
Date: 07/05/2019
Desc: Machine Learning application for the 'Predicting Molecular Properties' CHemistry And Mathematics
	  in Phase Space (CHAMPS) Kaggle competition.
	  Developed algorithm that can predict the magnetic interaction between
	  two atoms in a molecule (ie; "scalar_coupling_constant").
	  Predictive analytics with chemistry & chemical biology.
Ref: 
	- https://www.kaggle.com/todnewman/keras-neural-net-for-champs
	- https://www.kaggle.com/kabure/lightgbm-full-pipeline-model
	- https://www.kaggle.com/artgor/molecular-properties-eda-and-models
Ver: 
	- v1.0: working model 
	- v2.0: add more features, PCA, Pipelines, tweak RF hyperparameters
Youtube Video Link:
	- https://youtu.be/gXc-S-7TlyE
'''

# Dependancies

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]

from contextlib import contextmanager
import time
import gc
import os
import psutil
import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)



''' Global Variables  '''

# General variables 
ROOT_PATH = "C:/Users/rueen/Desktop/AI/kaggle_projects/CHAMPS_Scalar_Coupling_Constant_Prediction/champs-scalar-coupling/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
STRUCTURES_FILE = "structures.csv"
MULLIKEN_FILE = "mulliken_charges.csv"
MAGNETIC_FILE = "magnetic_shielding_tensors.csv"
SCALE_CONSTANTS_FILE = "scalar_coupling_contributions.csv"
POTENTIAL_ENERGY_FILE = "potential_energy.csv"
SUBMISSION_FILE = "sample_submission.csv"
PICKLE_FOLDER_PATH = ROOT_PATH+"PICKLES/"
TRAIN_PICKLE_FILENAME = "X_TRAIN_PICKLE.pickle"
TEST_PICKLE_FILENAME = "X_TEST_PICKLE.pickle"

# Hyperparameter variables

DIM_REDUCTION_PCA_THRESHOLD = 0.90 
SECONDARY_MODELS_KFOLD_CV_NUM = 3
# SECONDARY_MODELS_N_ESTIMATORS = [30, 40]
# SECONDARY_MODELS_MAX_FEATURES = [8, 16]
# SECONDARY_MODELS_MAX_DEPTH = [2, 6]

# Static variables

GOOD_COLUMNS_GBL = [
	'molecule_atom_index_0_dist_min',
	'molecule_atom_index_0_dist_max',
	'molecule_atom_index_1_dist_min',
	'molecule_atom_index_0_dist_mean',
	'molecule_atom_index_0_dist_std',
	'dist',
	'molecule_atom_index_1_dist_std',
	'molecule_atom_index_1_dist_max',
	'molecule_atom_index_1_dist_mean',
	'molecule_atom_index_0_dist_max_diff',
	'molecule_atom_index_0_dist_max_div',
	'molecule_atom_index_0_dist_std_diff',
	'molecule_atom_index_0_dist_std_div',
	'atom_0_couples_count',
	'molecule_atom_index_0_dist_min_div',
	'molecule_atom_index_1_dist_std_diff',
	'molecule_atom_index_0_dist_mean_div',
	'atom_1_couples_count',
	'molecule_atom_index_0_dist_mean_diff',
	'molecule_couples',
	'atom_index_1',
	'molecule_dist_mean',
	'molecule_atom_index_1_dist_max_diff',
	'molecule_atom_index_0_y_1_std',
	'molecule_atom_index_1_dist_mean_diff',
	'molecule_atom_index_1_dist_std_div',
	'molecule_atom_index_1_dist_mean_div',
	'molecule_atom_index_1_dist_min_diff',
	'molecule_atom_index_1_dist_min_div',
	'molecule_atom_index_1_dist_max_div',
	'molecule_atom_index_0_z_1_std',
	'y_0',
	'molecule_type_dist_std_diff',
	'molecule_atom_1_dist_min_diff',
	'molecule_atom_index_0_x_1_std',
	'molecule_dist_min',
	'molecule_atom_index_0_dist_min_diff',
	'molecule_atom_index_0_y_1_mean_diff',
	'molecule_type_dist_min',
	'molecule_atom_1_dist_min_div',
	'atom_index_0',
	'molecule_dist_max',
	'molecule_atom_1_dist_std_diff',
	'molecule_type_dist_max',
	'molecule_atom_index_0_y_1_max_diff',
	'molecule_type_0_dist_std_diff',
	'molecule_type_dist_mean_diff',
	'molecule_atom_1_dist_mean',
	'molecule_atom_index_0_y_1_mean_div',
	'molecule_type_dist_mean_div',
	'type'
	]

'''
Desc: Supporting method to reduces memory usage. 
''' 
def reduce_mem_usage(X_train, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = X_train.memory_usage().sum() / 1024**2
    for col in X_train.columns:
        col_type = X_train[col].dtypes
        if col_type in numerics:
            c_min = X_train[col].min()
            c_max = X_train[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    X_train[col] = X_train[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    X_train[col] = X_train[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    X_train[col] = X_train[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    X_train[col] = X_train[col].astype(np.int64)
            else:
                c_prec = X_train[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(np.float16).precision:
                    X_train[col] = X_train[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    X_train[col] = X_train[col].astype(np.float32)
                else:
                    X_train[col] = X_train[col].astype(np.float64)
    end_mem = X_train.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return X_train

'''
Desc: Calculates time duration
'''
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("\n{} - Duration = {:.0f} minutes".format(title, (time.time() - t0)/60))

'''
Desc: Displays current ram usage metric
'''
def show_ram_usage():
    py = psutil.Process(os.getpid())
    print("\nRAM utilization: {} GB\n".format(py.memory_info()[0]/2. ** 30))

'''
Desc: Maps atom's info
'''
def map_atom_info(df, atom_idx, df_structures):
    df = pd.merge(df, df_structures, how = 'left',
                  left_on  = ['molecule_name', 'atom_index_'+str(atom_idx)],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': 'atom_'+str(atom_idx),
                            'x': 'x_'+str(atom_idx),
                            'y': 'y_'+str(atom_idx),
                            'z': 'z_'+str(atom_idx)})
    return df

'''
Desc: Supporting method to help develop basic features
'''
def add_basic_features(X_train):
    X_train['molecule_couples'] = X_train.groupby('molecule_name')['id'].transform('count')
    X_train['molecule_dist_mean'] = X_train.groupby('molecule_name')['dist'].transform('mean')
    X_train['molecule_dist_min'] = X_train.groupby('molecule_name')['dist'].transform('min')
    X_train['molecule_dist_max'] = X_train.groupby('molecule_name')['dist'].transform('max')
    X_train['atom_0_couples_count'] = X_train.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    X_train['atom_1_couples_count'] = X_train.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')
    
    X_train['molecule_atom_index_0_x_1_std'] = X_train.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
    X_train['molecule_atom_index_0_y_1_mean'] = X_train.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
    X_train['molecule_atom_index_0_y_1_mean_diff'] = X_train['molecule_atom_index_0_y_1_mean'] - X_train['y_1']
    X_train['molecule_atom_index_0_y_1_mean_div'] = X_train['molecule_atom_index_0_y_1_mean'] / X_train['y_1']
    X_train['molecule_atom_index_0_y_1_max'] = X_train.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
    X_train['molecule_atom_index_0_y_1_max_diff'] = X_train['molecule_atom_index_0_y_1_max'] - X_train['y_1']
    X_train['molecule_atom_index_0_y_1_std'] = X_train.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
    X_train['molecule_atom_index_0_z_1_std'] = X_train.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')
    X_train['molecule_atom_index_0_dist_mean'] = X_train.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
    X_train['molecule_atom_index_0_dist_mean_diff'] = X_train['molecule_atom_index_0_dist_mean'] - X_train['dist']
    X_train['molecule_atom_index_0_dist_mean_div'] = X_train['molecule_atom_index_0_dist_mean'] / X_train['dist']
    X_train['molecule_atom_index_0_dist_max'] = X_train.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')
    X_train['molecule_atom_index_0_dist_max_diff'] = X_train['molecule_atom_index_0_dist_max'] - X_train['dist']
    X_train['molecule_atom_index_0_dist_max_div'] = X_train['molecule_atom_index_0_dist_max'] / X_train['dist']
    X_train['molecule_atom_index_0_dist_min'] = X_train.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
    X_train['molecule_atom_index_0_dist_min_diff'] = X_train['molecule_atom_index_0_dist_min'] - X_train['dist']
    X_train['molecule_atom_index_0_dist_min_div'] = X_train['molecule_atom_index_0_dist_min'] / X_train['dist']
    X_train['molecule_atom_index_0_dist_std'] = X_train.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')
    X_train['molecule_atom_index_0_dist_std_diff'] = X_train['molecule_atom_index_0_dist_std'] - X_train['dist']
    X_train['molecule_atom_index_0_dist_std_div'] = X_train['molecule_atom_index_0_dist_std'] / X_train['dist']
    X_train['molecule_atom_index_1_dist_mean'] = X_train.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
    X_train['molecule_atom_index_1_dist_mean_diff'] = X_train['molecule_atom_index_1_dist_mean'] - X_train['dist']
    X_train['molecule_atom_index_1_dist_mean_div'] = X_train['molecule_atom_index_1_dist_mean'] / X_train['dist']
    X_train['molecule_atom_index_1_dist_max'] = X_train.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')
    X_train['molecule_atom_index_1_dist_max_diff'] = X_train['molecule_atom_index_1_dist_max'] - X_train['dist']
    X_train['molecule_atom_index_1_dist_max_div'] = X_train['molecule_atom_index_1_dist_max'] / X_train['dist']
    X_train['molecule_atom_index_1_dist_min'] = X_train.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
    X_train['molecule_atom_index_1_dist_min_diff'] = X_train['molecule_atom_index_1_dist_min'] - X_train['dist']
    X_train['molecule_atom_index_1_dist_min_div'] = X_train['molecule_atom_index_1_dist_min'] / X_train['dist']
    X_train['molecule_atom_index_1_dist_std'] = X_train.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
    X_train['molecule_atom_index_1_dist_std_diff'] = X_train['molecule_atom_index_1_dist_std'] - X_train['dist']
    X_train['molecule_atom_index_1_dist_std_div'] = X_train['molecule_atom_index_1_dist_std'] / X_train['dist']
    X_train['molecule_atom_1_dist_mean'] = X_train.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')
    X_train['molecule_atom_1_dist_min'] = X_train.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')
    X_train['molecule_atom_1_dist_min_diff'] = X_train['molecule_atom_1_dist_min'] - X_train['dist']
    X_train['molecule_atom_1_dist_min_div'] = X_train['molecule_atom_1_dist_min'] / X_train['dist']
    X_train['molecule_atom_1_dist_std'] = X_train.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')
    X_train['molecule_atom_1_dist_std_diff'] = X_train['molecule_atom_1_dist_std'] - X_train['dist']
    X_train['molecule_type_0_dist_std'] = X_train.groupby(['molecule_name', 'type_0'])['dist'].transform('std')
    X_train['molecule_type_0_dist_std_diff'] = X_train['molecule_type_0_dist_std'] - X_train['dist']
    X_train['molecule_type_dist_mean'] = X_train.groupby(['molecule_name', 'type'])['dist'].transform('mean')
    X_train['molecule_type_dist_mean_diff'] = X_train['molecule_type_dist_mean'] - X_train['dist']
    X_train['molecule_type_dist_mean_div'] = X_train['molecule_type_dist_mean'] / X_train['dist']
    X_train['molecule_type_dist_max'] = X_train.groupby(['molecule_name', 'type'])['dist'].transform('max')
    X_train['molecule_type_dist_min'] = X_train.groupby(['molecule_name', 'type'])['dist'].transform('min')
    X_train['molecule_type_dist_std'] = X_train.groupby(['molecule_name', 'type'])['dist'].transform('std')
    X_train['molecule_type_dist_std_diff'] = X_train['molecule_type_dist_std'] - X_train['dist']

    X_train = reduce_mem_usage(X_train)
    return X_train

'''
Desc: Loading method 
'''
def loadData(X_train, X_test):
	# Load initial files 
	X_train = pd.read_csv(ROOT_PATH+TRAIN_FILE)
	X_test = pd.read_csv(ROOT_PATH+TEST_FILE)
	df_structures = pd.read_csv(ROOT_PATH+STRUCTURES_FILE)
	df_mulliken_charges = pd.read_csv(ROOT_PATH+MULLIKEN_FILE)
	df_magnetic_shielding_tensors = pd.read_csv(ROOT_PATH+MAGNETIC_FILE)
	df_scalar_constants = pd.read_csv(ROOT_PATH+SCALE_CONSTANTS_FILE)
	df_potential_energy = pd.read_csv(ROOT_PATH+POTENTIAL_ENERGY_FILE)
	df_submission = pd.read_csv(ROOT_PATH+SUBMISSION_FILE)

	# Merge in the scalar coupling contribution values, we will be predicting these, and adding them to our test set as test set does not have it like train set does...
	X_train = pd.merge(X_train, df_scalar_constants, how='left', left_on  = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],
                  right_on = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

	
	print("\n Output for X_train.head(3):\n ", X_train.head(3))
	print("\n Output for df_structures.head(3):\n ", df_structures.head(3))
	X_train = map_atom_info(X_train, 0, df_structures)
	X_train = map_atom_info(X_train, 1, df_structures)

	X_test = map_atom_info(X_test, 0, df_structures)
	X_test = map_atom_info(X_test, 1, df_structures)

	X_train = reduce_mem_usage(X_train)
	X_test= reduce_mem_usage(X_test)

	print("\nShape of X_train:", X_train.shape)
	print("\nShape of X_test:", X_test.shape)
	
	return X_train, X_test, df_submission

'''
Desc: Exploratory Data Analysis 
'''
def executeEDA(X_train, X_test):

	print("Number of training data rows = ", X_train.shape[0])
	print("Number of test data rows = ", X_test.shape[0])
	print("Number of unique molecules in train data = ", X_train['molecule_name'].nunique())
	print("Number of unique molecules in test data = ", X_test['molecule_name'].nunique())

	'''
	Test set is about 2 times smaller than train set
	'''


	# # Distribution of y target
	# X_train['scalar_coupling_constant'].plot(kind='hist', figsize=(20,5), bins=1000, title="Y Target distribution (scalar_coupling_constant)")
	# plt.show()

	# # Distribution of scalar coupling contributions (which make up the Y target value)
	# fig, ax = plt.subplots(2, 2, figsize=(20, 10))
	# X_train['fc'].plot(kind='hist', ax=ax.flat[0], bins=500, title='Fermi Contact contribution', color=color_pal[0])
	# X_train['sd'].plot(kind='hist', ax=ax.flat[1], bins=500, title='Spin-dipolar contribution', color=color_pal[1])
	# X_train['pso'].plot(kind='hist', ax=ax.flat[2], bins=500, title='Paramagnetic spin-orbit contribution', color=color_pal[2])
	# X_train['dso'].plot(kind='hist', ax=ax.flat[3], bins=500, title='Diamagnetic spin-orbit contribution', color=color_pal[3])
	# plt.show()

	# Swarmplot of the scalar coupling constant (y values) by type
	# sns.swarmplot(x='type', y='scalar_coupling_constant', data=X_train)
	# plt.show()
	
	# # Swarmplot of the fc feature by type
	# sns.swarmplot(x='type', y='fc', data=X_train)
	# plt.show()


	return

'''
Desc: Feature Engineering related logic 
'''
def featureEng(X_train, X_test):

	# Add the basics 
	train_p_0 = X_train[['x_0', 'y_0', 'z_0']].values
	train_p_1 = X_train[['x_1', 'y_1', 'z_1']].values
	test_p_0 = X_test[['x_0', 'y_0', 'z_0']].values
	test_p_1 = X_test[['x_1', 'y_1', 'z_1']].values
	X_train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
	X_test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
	X_train['dist_x'] = (X_train['x_0'] - X_train['x_1']) ** 2
	X_test['dist_x'] = (X_test['x_0'] - X_test['x_1']) ** 2
	X_train['dist_y'] = (X_train['y_0'] - X_train['y_1']) ** 2
	X_test['dist_y'] = (X_test['y_0'] - X_test['y_1']) ** 2
	X_train['dist_z'] = (X_train['z_0'] - X_train['z_1']) ** 2
	X_test['dist_z'] = (X_test['z_0'] - X_test['z_1']) ** 2
	X_train['type_0'] = X_train['type'].apply(lambda x: x[0])
	X_test['type_0'] = X_test['type'].apply(lambda x: x[0])
	X_train = add_basic_features(X_train)
	X_test = add_basic_features(X_test)



	
	return X_train, X_test

'''
Desc:
'''
def dimensionalityReduction(X_train):

	return X_train

'''
Desc: preProcessing method is responsible for any dataframe transformations to shape up the data for the model. 
'''
def preProcessing(X_train, X_test):

	GOOD_COLUMNS_GBL = [
	'molecule_atom_index_0_dist_min',
	'molecule_atom_index_0_dist_max',
	'molecule_atom_index_1_dist_min',
	'molecule_atom_index_0_dist_mean',
	'molecule_atom_index_0_dist_std',
	'dist',
	'molecule_atom_index_1_dist_std',
	'molecule_atom_index_1_dist_max',
	'molecule_atom_index_1_dist_mean',
	'molecule_atom_index_0_dist_max_diff',
	'molecule_atom_index_0_dist_max_div',
	'molecule_atom_index_0_dist_std_diff',
	'molecule_atom_index_0_dist_std_div',
	'atom_0_couples_count',
	'molecule_atom_index_0_dist_min_div',
	'molecule_atom_index_1_dist_std_diff',
	'molecule_atom_index_0_dist_mean_div',
	'atom_1_couples_count',
	'molecule_atom_index_0_dist_mean_diff',
	'molecule_couples',
	'atom_index_1',
	'molecule_dist_mean',
	'molecule_atom_index_1_dist_max_diff',
	'molecule_atom_index_0_y_1_std',
	'molecule_atom_index_1_dist_mean_diff',
	'molecule_atom_index_1_dist_std_div',
	'molecule_atom_index_1_dist_mean_div',
	'molecule_atom_index_1_dist_min_diff',
	'molecule_atom_index_1_dist_min_div',
	'molecule_atom_index_1_dist_max_div',
	'molecule_atom_index_0_z_1_std',
	'y_0',
	'molecule_type_dist_std_diff',
	'molecule_atom_1_dist_min_diff',
	'molecule_atom_index_0_x_1_std',
	'molecule_dist_min',
	'molecule_atom_index_0_dist_min_diff',
	'molecule_atom_index_0_y_1_mean_diff',
	'molecule_type_dist_min',
	'molecule_atom_1_dist_min_div',
	'atom_index_0',
	'molecule_dist_max',
	'molecule_atom_1_dist_std_diff',
	'molecule_type_dist_max',
	'molecule_atom_index_0_y_1_max_diff',
	'molecule_type_0_dist_std_diff',
	'molecule_type_dist_mean_diff',
	'molecule_atom_1_dist_mean',
	'molecule_atom_index_0_y_1_mean_div',
	'molecule_type_dist_mean_div',
	'type'
	]

	for f in ['atom_1', 'type_0', 'type']:
	    if f in GOOD_COLUMNS_GBL:
	    	lbl = LabelEncoder()
	    	lbl.fit(list(X_train[f].values) + list(X_test[f].values))
	    	X_train[f] = lbl.transform(list(X_train[f].values))
	    	X_test[f] = lbl.transform(list(X_test[f].values))

	
	
	X_train = X_train.drop("molecule_name", axis=1)
	X_test = X_test.drop("molecule_name", axis=1)

	#X_train = X_train.reset_index()
	#X_test = X_test.reset_index()

 	# Preprocessing dataframe for np.Inf (Infinity values) to resolve any ValueError
	pd.options.mode.use_inf_as_na = True   # Worked to resolve ValueError
	for col in range(0,len(GOOD_COLUMNS_GBL)):
		X_train[GOOD_COLUMNS_GBL[col]].fillna(X_train[GOOD_COLUMNS_GBL[col]].max(),inplace=True)


	# print("\n Creating PICKLE (.pickle) files for X_train and X_test files...")
	# X_tr_PICKLE = open(PICKLE_FOLDER_PATH + TRAIN_PICKLE_FILENAME, "wb")
	# pickle.dump(X_train, X_tr_PICKLE, protocol=4)
	# X_tr_PICKLE.close()
	# X_test_PICKLE = open(PICKLE_FOLDER_PATH + TEST_PICKLE_FILENAME, "wb")
	# pickle.dump(X_test, X_test_PICKLE, protocol=4)
	# X_test_PICKLE.close()



	return X_train, X_test


'''
Desc: Support method to show model evaluation metrics after the K Fold Cross Validation (CV) is done.
'''
def display_scores(scores):
	print("\n K-Fold Cross Validation Scores: ", scores)
	print("\n K-Fold Cross Validation Mean: ", scores.mean())
	print("\n K-Fold Cross Validation Standard Deviation: ", scores.std())


'''
Desc: All models (main or otherwise) are developed in this method. 
	  Predict the scalar coupling contribution & other molecular property values missing and use in final dataframe for final model. 
'''
def createModel(X_train, X_test):

	'''
	Model Development notes:
	The strategy is to create one model for each molecule type. 
	Before we can do that, we need to predict the scalar coupling contribution
	values for "fc", "pso", "dso", and "sd" and put it in X_test. 
	Additionally, need to predict the Mulliken charge & potential energy values as well.
	'''

	# Retrieve the PICKLE files for the X_train and the X_test.
	# train_PICKLE = open(PICKLE_FOLDER_PATH + TRAIN_PICKLE_FILENAME, "rb")
	# X_train = pickle.load(train_PICKLE)
	# test_PICKLE = open(PICKLE_FOLDER_PATH + TEST_PICKLE_FILENAME, "rb")
	# X_test = pickle.load(test_PICKLE)

	# X_train_fc = X_train[GOOD_COLUMNS_GBL].copy()
	# X_test = X_test[GOOD_COLUMNS_GBL].copy()
	# y_train = X_train['scalar_coupling_constant']

	
	'''
	Create models for predicting fc, sd, pso, dso features (Scalar coupling contributions)
	'''

	# SECONDARY_MODELS_PARAM_GRID = [
	# 			 {"n_estimators":SECONDARY_MODELS_N_ESTIMATORS},
	# 			 {"n_estimators":SECONDARY_MODELS_N_ESTIMATORS, "max_features":SECONDARY_MODELS_MAX_FEATURES},
	# 			 {"n_estimators":SECONDARY_MODELS_N_ESTIMATORS, "max_depth":SECONDARY_MODELS_MAX_DEPTH},
	# 			 {"n_estimators":SECONDARY_MODELS_N_ESTIMATORS, "max_depth":SECONDARY_MODELS_MAX_DEPTH, "max_features":SECONDARY_MODELS_MAX_FEATURES},
	# 			 ]

	print("\n Commencing Random Forest Regressor Ensemble...")

	X_test = pd.DataFrame(X_test)
	X_train = pd.DataFrame(X_train)
	X_train_fc = pd.DataFrame(X_train_fc)

	print("\n X_train_fc describe: ", X_train_fc.describe())

	# main_pipe = Pipeline([
	# 			('simple_imputer', SimpleImputer(strategy="median")),
 #                ('std_scaler', StandardScaler()),
 #                ('pca', PCA(n_components=DIM_REDUCTION_PCA_THRESHOLD)),
 #                ])

	main_pipe = Pipeline([
				('simple_imputer', SimpleImputer(strategy="median")),
                ('std_scaler', StandardScaler()),
                ])

	X_train_fc = main_pipe.fit_transform(X_train_fc)

	X_train_fc = pd.DataFrame(X_train_fc)
	X_test = pd.DataFrame(X_test)

	# Scalar coupling constants
	y_train_fc = X_train['fc']
	y_train_sd = X_train['sd']
	y_train_pso = X_train['pso']
	y_train_dso = X_train['dso']


	print("\n Commencing RFRegressor for Target = 'FC'...")
	# Using RandomForestRegressor Ensemble Model
	fc_random_forest_regressor = RandomForestRegressor(n_estimators=20,
														max_leaf_nodes=3,
														max_depth=6,
														max_features=7,
														n_jobs=-1,
														bootstrap=True,
														random_state=42,
														verbose=3,
														criterion='mse',
														oob_score=True)

    # Train it
	fc_random_forest_regressor.fit(X_train_fc, y_train_fc)
	print(" \n FC RF Fitting completed.....")
	''' Create & Evaluate the Random Forest Regressor Models '''
	print(" \n FC RF predictions commenced.....")
	fc_predictions = fc_random_forest_regressor.predict(X_train_fc)
	tree_mse = mean_squared_error(y_train_fc, fc_predictions)
	tree_mse = np.sqrt(tree_mse)
	print("\n Feature: FC - RandomForestRegressor RMSE Score: ", tree_mse)
	print("\n Feature: FC - RandomForestRegressor OOB Score: ", fc_random_forest_regressor.oob_score_)
	print("\n Feature: FC - RandomForestRegressor MAE Score: ", mean_absolute_error(y_train_fc, fc_predictions))

	print("\n Feature: FC - RandomForestRegressor using K Fold CV - cross_val_score()...")
	fc_random_forest_regressor_scores = cross_val_score(fc_random_forest_regressor, X_train_fc, y_train_fc,
													 scoring="neg_mean_squared_error",
													 cv=SECONDARY_MODELS_KFOLD_CV_NUM,
													 verbose=1)
	fc_random_forest_regressor_rmse_scores = np.sqrt(-fc_random_forest_regressor_scores)
	display_scores(fc_random_forest_regressor_rmse_scores)


	print("\n Commencing RFRegressor for Target = 'SD'...")

	sd_random_forest_regressor = RandomForestRegressor(n_estimators=7,
														max_leaf_nodes=3,
														max_depth=8,
														max_features=5,
														n_jobs=-1,
														bootstrap=True,
														random_state=42,
														verbose=2,
														criterion='mse',
														oob_score=True)


	# Train it
	sd_random_forest_regressor.fit(X_train_fc, y_train_sd)
	''' Create & Evaluate the Random Forest Regressor Models '''
	sd_predictions = sd_random_forest_regressor.predict(X_train_fc)
	tree_mse = mean_squared_error(y_train_sd, sd_predictions)
	tree_mse = np.sqrt(tree_mse)
	print("\n Feature: SD - RandomForestRegressor RMSE Score: ", tree_mse)
	#print("\n Feature: SD - RandomForestRegressor MAE Score: ", mean_absolute_error(y_train_fc, sd_predictions))
	print("\n Feature: SD - RandomForestRegressor OOB Score: ", sd_random_forest_regressor.oob_score_)
	#print("\n Feature: FC - RandomForestRegressor MAE Score: ", mean_absolute_error(y_train_fc, sd_predictions))
	# print("\n Feature: SD - RandomForestRegressor using K Fold CV - cross_val_score()...")
	# sd_random_forest_regressor_scores = cross_val_score(sd_random_forest_regressor, X_train_fc, y_train_sd,
	# 												 scoring="neg_mean_squared_error",
	# 												 cv=SECONDARY_MODELS_KFOLD_CV_NUM,
	# 												 verbose=1)
	# sd_random_forest_regressor_rmse_scores = np.sqrt(-sd_random_forest_regressor_scores)
	# display_scores(sd_random_forest_regressor_rmse_scores)

	

	print("\n Commencing RFRegressor for Target = 'PSO'...")

	pso_random_forest_regressor = RandomForestRegressor(n_estimators=7,
														max_leaf_nodes=3,
														max_features=5,
														max_depth=5,
														n_jobs=-1,
														bootstrap=True,
														random_state=42,
														verbose=2,
														criterion='mse',
														oob_score=True)

    # Train it
	pso_random_forest_regressor.fit(X_train_fc, y_train_pso)
	''' Create & Evaluate the Random Forest Regressor Models '''
	pso_predictions = pso_random_forest_regressor.predict(X_train_fc)
	tree_mse = mean_squared_error(y_train_pso, pso_predictions)
	tree_mse = np.sqrt(tree_mse)
	print("\n Feature: PSO - RandomForestRegressor RMSE Score: ", tree_mse)
	#print("\n Feature: PSO - RandomForestRegressor MAE Score: ", mean_absolute_error(y_train_fc, pso_predictions))
	print("\n Feature: PSO - RandomForestRegressor OOB Score: ", pso_random_forest_regressor.oob_score_)
	#print("\n Feature: FC - RandomForestRegressor MAE Score: ", mean_absolute_error(y_train_fc, pso_predictions))
	# print("\n Feature: PSO - RandomForestRegressor using K Fold CV - cross_val_score()...")
	# pso_random_forest_regressor_scores = cross_val_score(pso_random_forest_regressor, X_train_fc, y_train_pso,
	# 												 scoring="neg_mean_squared_error",
	# 												 cv=SECONDARY_MODELS_KFOLD_CV_NUM,
	# 												 verbose=1)
	# pso_random_forest_regressor_rmse_scores = np.sqrt(-pso_random_forest_regressor_scores)
	# display_scores(pso_random_forest_regressor_rmse_scores)

	


	print("\n Commencing RFRegressor for Target = 'DSO'...")
	dso_random_forest_regressor = RandomForestRegressor(n_estimators=5,
														max_leaf_nodes=3,
														max_depth=5,
														n_jobs=-1,
														bootstrap=True,
														random_state=42,
														verbose=2,
														criterion='mse',
														oob_score=True)

    # Train it
	dso_random_forest_regressor.fit(X_train_fc, y_train_dso)
	''' Create & Evaluate the Random Forest Regressor Models '''
	dso_predictions = dso_random_forest_regressor.predict(X_train_fc)
	tree_mse = mean_squared_error(y_train_dso, dso_predictions)
	tree_mse = np.sqrt(tree_mse)
	print("\n Feature: DSO - RandomForestRegressor RMSE Score: ", tree_mse)
	#print("\n Feature: DSO - RandomForestRegressor MAE Score: ", mean_absolute_error(y_train_fc, dso_predictions))
	print("\n Feature: DSO - RandomForestRegressor OOB Score: ", dso_random_forest_regressor.oob_score_)
	#print("\n Feature: FC - RandomForestRegressor MAE Score: ", mean_absolute_error(y_train_fc, dso_predictions))

	# print("\n Feature: DSO - RandomForestRegressor using K Fold CV - cross_val_score()...")
	# dso_random_forest_regressor_scores = cross_val_score(dso_random_forest_regressor, X_train_fc, y_train_dso,
	# 												 scoring="neg_mean_squared_error",
	# 												 cv=SECONDARY_MODELS_KFOLD_CV_NUM,
	# 												 verbose=1)
	# dso_random_forest_regressor_rmse_scores = np.sqrt(-dso_random_forest_regressor_scores)
	# display_scores(dso_random_forest_regressor_rmse_scores)

	pd.options.mode.use_inf_as_na = True   # Worked to resolve ValueError
	X_test = main_pipe.fit_transform(X_test)
	X_test = pd.DataFrame(X_test)


	df_temp_test = pd.DataFrame()
	df_temp_X_train_fc = pd.DataFrame()

	df_temp_test['fc'] = fc_random_forest_regressor.predict(X_test)				# Add missing feature to dataframe	
	df_temp_X_train_fc['fc'] = fc_random_forest_regressor.predict(X_train_fc)	# Add missing feature to dataframe

	df_temp_test['sd'] = sd_random_forest_regressor.predict(X_test)				# Add missing feature to dataframe
	df_temp_X_train_fc['sd'] = sd_random_forest_regressor.predict(X_train_fc)	# Add missing feature to dataframe

	df_temp_test['pso'] = pso_random_forest_regressor.predict(X_test)			# Add missing feature to dataframe
	df_temp_X_train_fc['pso'] = pso_random_forest_regressor.predict(X_train_fc)	# Add missing feature to dataframe

	df_temp_test['dso'] = dso_random_forest_regressor.predict(X_test)			# Add missing feature to dataframe
	df_temp_X_train_fc['dso'] = dso_random_forest_regressor.predict(X_train_fc)	# Add missing feature to dataframe

	

	X_test['fc'] = df_temp_test['fc']
	X_test['sd'] = df_temp_test['sd']
	X_test['pso'] = df_temp_test['pso']
	X_test['dso'] = df_temp_test['dso']
	X_train_fc['fc'] = df_temp_X_train_fc['fc']
	X_train_fc['sd'] = df_temp_X_train_fc['sd']`
	X_train_fc['pso'] = df_temp_X_train_fc['pso']
	X_train_fc['dso'] = df_temp_X_train_fc['dso']

	X_train_fc = main_pipe.fit_transform(X_train_fc)	
	X_train_fc = pd.DataFrame(X_train_fc)
	X_test = main_pipe.fit_transform(X_test)	
	X_test = pd.DataFrame(X_test)

	# Final Model Development 
	print("\n Number of X_train_fc columns: ", len(X_train_fc.columns))
	print("\n Number of X_test columns: ", len(X_test.columns))
	print("\n Commencing FINAL MODEL RFRegressor for Target = 'SCALAR_COUPLING_CONSTANT'...")
	
	print("\n Type of X_train_fc: ", type(X_train_fc))

	print("\n X_train_fc # columns: ", len(X_train_fc.columns))
	print("\n X_test # columns: ", len(X_test.columns))

	FINAL_MODEL_random_forest_regressor = RandomForestRegressor(n_estimators=2,
														max_leaf_nodes=4,
														max_depth=2,
														max_features=2,
														n_jobs=-1,
														bootstrap=True,
														random_state=42,
														verbose=1,
														criterion='mse',
														oob_score=True)
	
	
	

	 
	
	FINAL_MODEL_random_forest_regressor.fit(X_train_fc, y_train)




	''' Create & Evaluate the Random Forest Regressor Model '''
	FINAL_predictions = FINAL_MODEL_random_forest_regressor.predict(X_test)


	print("\n Feature: FINAL MODEL - RandomForestRegressor using K Fold CV - cross_val_score()...")
	FINAL_MODEL_random_forest_regressor_scores = cross_val_score(FINAL_MODEL_random_forest_regressor, X_train_fc, y_train,
													 scoring="neg_mean_squared_error",
													 cv=SECONDARY_MODELS_KFOLD_CV_NUM,
													 verbose=1)
	FINAL_MODEL_random_forest_regressor_rmse_scores = np.sqrt(-FINAL_MODEL_random_forest_regressor_scores)
	display_scores(FINAL_MODEL_random_forest_regressor_rmse_scores)

	# Feature Importances
	for name, score in zip(X_train_fc.columns, FINAL_MODEL_random_forest_regressor.feature_importances_):
		print(name, score)

	# Save the model
	final_model_filename = "MODEL_Low_{}".format(int(time.time()))
	pickle.dump(FINAL_MODEL_random_forest_regressor, open(final_model_filename, 'wb'))



	# feats = {}
	# for feature, importance in zip(X_train.columns, FINAL_MODEL_random_forest_regressor.feature_importances_):
	# 	feats[feature] = importance
	# importances = pd.DataFrame.from_dict(feats, orient="index").rename(columns={0:'Gini-importance'})
	# importances.sort_values(by="Gini-importance").plot(kind='bar', rot=45)
	# plt.show()

	GOOD_COLUMNS_GBL.append("FC")
	GOOD_COLUMNS_GBL.append("SD")
	GOOD_COLUMNS_GBL.append("PSO")
	GOOD_COLUMNS_GBL.append("DSO")

	# Feature Importances in matplotlib
	#Importance = pd.DataFrame(index=X_train_fc.columns)
	Importance = pd.DataFrame(index=GOOD_COLUMNS_GBL)
	Importance['Importance'] = FINAL_MODEL_random_forest_regressor.feature_importances_*100
	Importance.loc[Importance['Importance'] > 1.5].sort_values('Importance').head(40).plot(kind='barh', figsize=(14,28), color='r', title="Feature Importance")
	plt.xlabel("Level")
	plt.show()




	return FINAL_MODEL_random_forest_regressor, X_test

'''
Desc: Predict & Evaluate 
'''
def predict(model):
	return y_pred

'''
Desc: Export results and prepare .csv file for submission per requirements.
'''
def exportEvaluationResults(model, df_submission, X_test):
	print("\n Exporting prediction results to submission.csv .....")
	df_submission['scalar_coupling_constant'] = model.predict(X_test)
	df_submission.to_csv('submission.csv', index=False)
	print("\n", df_submission.head(5)) 
	return


'''
Desc: Application's main method
'''
def main():
	X_train = pd.DataFrame()
	X_test = pd.DataFrame()
	X_test_final = pd.DataFrame()

	#df_submission = pd.read_csv(ROOT_PATH+SUBMISSION_FILE)
	df_submission = pd.DataFrame()   # Uncomment if running full script from data load too. 

	with timer(" Data Loading:"):
		print("\n Data Loading...")
		X_train, X_test, df_submission = loadData(X_train, X_test)
		print("\n Shape of X_train:", X_train.shape)
		print("\n Shape of X_test:", X_test.shape)
		show_ram_usage()
		print("\n Output for X_train.head(3): ", X_train.head(3))
		gc.collect()

	with timer(" Exploratory Data Analysis:"):
		print("\n Exploratory Data Analysis...")
		executeEDA(X_train, X_test)

	with timer(" Feature Engineering:"):
		print("\n Feature Engineering...")
		X_train, X_test = featureEng(X_train, X_test)
		print("\n Shape of X_train:", X_train.shape)
		print("\n Shape of X_test:", X_test.shape)
		show_ram_usage()
		print(X_train.head(3))
		gc.collect()

	with timer(" Preprocessing:"):
		print("\n Preprocessing...")
		X_train, X_test = preProcessing(X_train, X_test)
		print("\n Shape of X_train:", X_train.shape)
		print("\n Shape of X_test:", X_test.shape)
		show_ram_usage()
		print(X_train.head(3))
		gc.collect()

	with timer(" Dimensionality Reduction:"):
		print("\n Dimensionality Reduction...")
		#X_train = dimensionalityReduction(X_train)
		show_ram_usage()
		gc.collect()

	with timer(" Model Development & Evaluation:"):
		print("\n Model Development & Evaluation...")
		prospect_model, X_test_final = createModel(X_train, X_test)
		show_ram_usage()
		gc.collect()

	with timer(" Results EDA:"):
		print("\n Results EDA...")
		#resultsEDA()
		gc.collect()

	with timer(" Export Evaluation Results:"):
		print("\n Export Evaluation Results...")
		exportEvaluationResults(prospect_model, df_submission, X_test_final)
		show_ram_usage()
		gc.collect()



if __name__ == "__main__":
	with timer("\n -*- Complete CHAMPS ML Application -*-"):
		print("\n -*- Opening CHAMPS ML Application -*-")
		main()
		show_ram_usage()
		print("\n -*- Exiting CHAMPS ML Application -*-")

