"""Data_Preparation_HomoSapiens
This program reads in the annotations and the expressions of the genes (mouse)
After reading in, the twod dataframes are being merged and preprocessed
The label vectors for every gene example is also generated
"""

# Imports 
#def import_or_install(package):
#    try:
#        __import__(package)
#    except ImportError:
#      pip.main(['install', package])

#import_or_install('sklearn')
#import_or_install('numpy')
#import_or_install('pandas')
#import_or_install('tensorflow')

import warnings
warnings.filterwarnings('ignore')
# os
import os
from os import path
import sys
# Data Preparation
import pandas as pd
import numpy
import matplotlib.pyplot as plt
# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Preprocessing 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
#from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
import random

print('All needed packages imported')

# Initiate the Data Preparation and define the paths to the needed data
expressionsPath = 'Datasets/HomoSapiensExpressions300.csv'
annotationsPath = 'Datasets/HomoSapiensAnnotations.csv'
panthersGOPath = 'Datasets/PANTHERGOslim.obo.txt'

# Create Data_Analysis folder
if not os.path.exists('Data_Analysis'):
    os.makedirs('Data_Analysis')
# Create Models folder
if not os.path.exists('Models'):
    os.makedirs('Models')
# Create Evaluation folder
if not os.path.exists('Evaluation'):
    os.makedirs('Evaluation')
# Create Predictions folder
if not os.path.exists('Predictions'):
        os.makedirs('Predictions')

class Data_Preparation():
  df, dfNoPC, annoNoPC = None, None, None
  attributes, classes = [], []

  species = 'HomoSapiens'
  functionPrediction = 'molecular_function'
  primaryKey = 'geneId'
  expr_analysis = False

  def __init__(self, expressions_path, annotations_path, panthers_goslim_path):
    self.expressions_path = expressions_path
    self.annotations_path = annotations_path
    self.panthers_goslim_path = panthers_goslim_path

  def construct_gene_dataset(self):
    # Read in the annotations
    print('Read in the annotations for '+self.species) 
    dfAnno = pd.read_csv(self.annotations_path)
    try:
        dfAnno = dfAnno.drop(['Unnamed: 0'], axis=1)
        dfAnno = dfAnno.rename(columns={"X": self.primaryKey})
    except:
        pass
    try:
        dfAnno = dfAnno.drop(['goslim_goa_accession'], axis=1)
    except:
        pass
    print(self.species+' Annotations with shape:' + str(dfAnno.shape))
    # Read in the expressions
    print('Read in the expressions for '+self.species)       
    dfExpr = pd.read_csv(self.expressions_path)
    try:
        dfExpr = dfExpr.drop(['Unnamed: 0'], axis=1)
        dfExpr = dfExpr.rename(columns={"X": self.primaryKey})
    except:
        pass
    # Discard the version in the geneIds
    sep = '.'
    try:
      dfExpr[self.primaryKey] = dfExpr[self.primaryKey].apply(lambda x: x.split(sep, 1)[0])
    except:
      pass
    print(self.species+' Expressions with shape:' + str(dfExpr.shape))

    # Shift the expressions to a positive area
    # Retrive the biggest negative number
    lowestValue = min(list(dfExpr.loc[:, dfExpr.columns != self.primaryKey].min()))
    print('Lowest detected value to shift with: '+str(lowestValue))
    # Shift by the biggest negative number
    dfExpr.loc[:, dfExpr.columns != self.primaryKey] = dfExpr.loc[:, dfExpr.columns != self.primaryKey]+abs(lowestValue)*(1/2)
    
    # Merge the expressions and annotations for our final dataset
    print('Merge the annotations and expressions')
    self.df = pd.merge(dfAnno, dfExpr, on=self.primaryKey)
    self.dfNoPC = pd.merge(dfExpr, dfAnno, on=self.primaryKey)
    print(self.species+' dataset with shape:' + str(self.df.shape))
    print('Extract the non protein coding genes and save in dfNoPC')
    # Define the non protein coding genes
    self.dfNoPC = pd.merge(dfExpr, dfAnno, on=[self.primaryKey], how="left", indicator=True).query('_merge=="left_only"')

  def construct_datasets(self):

    # Build the real dataset
    self.construct_gene_dataset()

  def clean_dataset(self):
    print('Initiate preprocessing and cleaning')
    # Check for null values in the annotated dataset
    True in self.df.isnull().values
    self.df.isnull().sum()
    # Replace nan-values with ''
    self.df = self.df.replace(numpy.nan, '', regex=True)
    # Check for null values
    True in self.df.isnull().values
    # Same procedure for the non protein coding genes
    # Replace nan-values with ''
    self.dfNoPC = self.dfNoPC.replace(numpy.nan, '', regex=True)
    # Check for null values
    True in self.df.isnull().values
    print('Data cleaning finished')

  def prepare_data(self):
    print('Extract the features for '+self.species)

    # Extract the features
    attributes = list(self.df.columns)
    attributes.remove('goslim_goa_description')
    attributes.remove('geneId')
    try:
        attributes.remove('gene_biotype')    
    except:
        pass
    try:
        attributes.remove('goslim_goa_accession')
    except:
        pass
    try:
        attributes.remove('goslim')
    except:
        pass
    self.attributes = attributes
    print(str(len(self.attributes))+' features in '+self.species)

    # Build up the label matrix
    print('Build the class matrix')
    classDistr = {}
    classDistrDf = pd.DataFrame()
    for i in self.df.index:
        currentRow = self.df.loc[i]['goslim'].split(', ')
        for molFunc in currentRow:
            molFunc = molFunc.lstrip().rstrip()
            self.df.at[i, molFunc] = 1
            if(molFunc not in classDistr):
                classDistr[molFunc] = 1
            else:
                classDistr[molFunc] = classDistr[molFunc] + 1 

    # Replace every NaN with zero        
    self.df[list(classDistr.keys())] = self.df[list(classDistr.keys())].fillna(0)
  
    # Define which are the class columns
    classes = list(classDistr.keys())
    # Discard "molecular function" as class
    try:
      classes.remove(self.functionPrediction)
    except:
      pass
    try:
      classDistr.pop(self.functionPrediction)
    except:
      pass
    print(str(len(classes))+' classes detected in '+self.species)

    print('Extract annotated non-protein coding genes and save in annoNoPC')
    # Extract annotated genes which are not protein coding 
    self.annoNoPC = self.df[self.df['gene_biotype']!='protein_coding']
    # Idea: use this annotated/labeled non-protein coding genes as our labeled target domain
    print('annoNoPC shape: '+str(self.annoNoPC.shape))
    # Exclude these polymorphic_pseudogene in our source domain
    self.df = self.df[self.df['gene_biotype']!='polymorphic_pseudogene']

    # Plot the current class distribution
    fig = plt.subplots(figsize=(34,10))
    self.df[classes].sum().plot(kind='bar')
    plt.savefig('Data_Analysis/'+self.species+'AllClasses.png', bbox_inches = "tight", dpi=300)
    print(str(len(classes))+' overall classes in '+self.species)
    # Define the minimal number of examples for each class
    minExamples = 200
    # Remove the very underrepresented classes
    newClassDistr = classDistr.copy()
    for (key, value) in classDistr.items():
      # Check if key is even then add pair to new dictionary
      if (value < minExamples):
        newClassDistr.pop(key.lstrip().rstrip())
    newClasses = list(newClassDistr.keys())
    print(str(len(newClasses))+' classes remain after discarding classes with less than '+str(minExamples)+' examples')
    # Drop the classes
    try:
      self.df = self.df.drop(classesToDrop, axis=1)
      #self.dfShuffled = self.dfShuffled.drop(classesToDrop, axis=1)
    except:
      print('Error in dropping')
      pass
    # If reduction to GOslim 
    print('Reduce the classes on GOslims from Panther')
    goslims = []
    goslimFile=open(self.panthers_goslim_path)
    lines=goslimFile.readlines()
    for line in range(0, len(lines)):
      if lines[line].strip() == '[Term]':
            # Extract the next 3 attributes (Id, name, namespace)
            goslimId = lines[line+1].strip().split(' ')[1]
            goslim = lines[line+2].strip()[6:].replace(",", " and")
            goslimSpace = lines[line+3].strip().split(' ')[1]
            if (goslimSpace.strip()==self.functionPrediction):
              # Store the goslim
              goslims.append(goslim)
    newClasses = list(set(goslims) & set(newClasses))
    self.classes = newClasses
    print('After reducing to Panthers GOslims, '+str(len(self.classes))+' classes remain')
    # Plot the new class distribution
    fig = plt.subplots(figsize=(34,10))
    self.df[self.classes].sum().plot(kind='bar', fontsize=20)
    plt.savefig('Data_Analysis/'+self.species+'ClassesTrimmed.png', bbox_inches = "tight", dpi=300)
    # ML-SMOTE approach has been excluded exluced!
    # Class weights approach has been excluded!
    
  def export_datasets(self):
    # Create ML_Datasets folder
    if not os.path.exists('ML_Datasets'):
        os.makedirs('ML_Datasets')
    # Dataset - protein coding genes to learn with
    self.df.to_csv(r'ML_Datasets/PCgenes.csv')
    # Dataset - annotated non-protein coding genes to test the models
    self.annoNoPC.to_csv(r'ML_Datasets/annoNPCgenes.csv')
    # Dataset - lncRNA samples
    self.dfNoPC.to_csv(r'ML_Datasets/lncRNAs.csv')
    # Export the attributes and classnames
    with open('ML_Datasets/classes.txt', 'w') as f:
        for item in self.classes:
            f.write("%s\n" % item)
    # Export the attributes and classnames
    with open('ML_Datasets/attributes.txt', 'w') as f:
        for item in self.attributes:
            f.write("%s\n" % item)

# Instantiate an object
data_Prep = Data_Preparation(expressionsPath, annotationsPath, panthersGOPath)

# Read in the data and construct the datasets
data_Prep.construct_datasets()

# Clean the dataset
data_Prep.clean_dataset()

# Initiate the feature extraction and class vector generation
data_Prep.prepare_data()

# Export the datasets
data_Prep.export_datasets()
