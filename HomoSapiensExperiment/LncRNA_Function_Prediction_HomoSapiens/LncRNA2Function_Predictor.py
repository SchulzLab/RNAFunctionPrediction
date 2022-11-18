import warnings
warnings.filterwarnings('ignore')
# os
import os
import sys
from os import path

# Data Preparation
import pandas as pd
import numpy
from tabulate import tabulate
import math

# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Preprocessing 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
#from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MultiLabelBinarizer

# Keras framework
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import plot_model

# Neural Network
from numpy import std
from sklearn.model_selection import RepeatedKFold
#from keras.models import Sequential
#from keras.layers import Dense

# sklearn classifier models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# save models
import pickle
# plot
import matplotlib.pyplot as plt
#import seaborn as sns
# sklearn metrics
from sklearn import metrics
import statistics

# model selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

print('All needed packages got imported')

"""# Properties"""

species = 'HomoSapiens'
functionPrediction = 'molecular_function'
primaryKey = 'geneId'
#modelUser = sys.argv[1]
print('LncRNA2Function_Predictor initiated')

"""# Read in the datasets"""

# Dataset - protein coding genes to learn with
df = pd.read_csv(r'ML_Datasets/PCgenes.csv', index_col=[0])
# Dataset - annotated non-protein coding genes to test the models
annoNoPC = pd.read_csv(r'ML_Datasets/annoNPCgenes.csv', index_col=[0])
# Dataset - lncRNA samples
dfNoPC = pd.read_csv(r'ML_Datasets/lncRNAs.csv', index_col=[0])

# Extract the genes of interests
genes_of_interest = pd.read_csv('Datasets/lncRNAsHomoSapiens.txt', delimiter = "\t")
# Discard the version in the geneIds
sep = '.'
genes_of_interest['Gene ID'] = genes_of_interest['Gene ID'].apply(lambda x: x.split(sep, 1)[0])

# Extract the expression values of the genes of interest
genes_of_interest = dfNoPC[dfNoPC[primaryKey].isin(genes_of_interest['Gene ID'])]
# TODO use all genes for prediction
genes_of_interest = dfNoPC

print('Expressions for the genes of interest extracted')
print(genes_of_interest.shape)

"""# Read in the class and attribute names"""

# Read in the class and attribute names
with open('ML_Datasets/classes.txt') as classesFile:
    classes = classesFile.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
classes = [x.strip() for x in classes] 

# Read in the class and attribute names
with open('ML_Datasets/attributes.txt') as attributesFile:
    attributes = attributesFile.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
attributes = [x.strip() for x in attributes]

"""# Read in the models"""
# Read in the compressed models 
# Read in the MLP
mlp = load_model('Models/ML_Approaches/Sequential_'+species+'.h5')
# Read in the mlknn
mlknn = pickle.load(open('Models/ML_Approaches/MLknn_'+species+'.sav', 'rb'))
# Read in the RandomForest
rf = pickle.load(open('Models/ML_Approaches/RandomForest_'+species+'.sav', 'rb'))
# Read in the chain of knns
chain_knn = pickle.load(open('Models/Chain_Approaches/Chain_knn_'+species+'.sav', 'rb'))
# Read in the chain of rfs
chain_rf = pickle.load(open('Models/Chain_Approaches/Chain_RandomForest_'+species+'.sav', 'rb'))
# Read in the chain of CNNs and define methods for the prediction
# Construct a map label --> CNN
labelCNNMap = {}
historyPath = 'Evaluation/Chain_Approaches/CNN/trainHistoryDict-CNN-'
# Do it for every class:

#TODO: Remove
for label in classes[0:5]:
  # Try to load the existing classifier
  cnn = load_model('Models/Chain_Approaches/CNN/CNN_'+label+'.h5')
  print('CNN_'+label+'.h5 ' +'found and loaded')
  # Save in the map
  labelCNNMap[label]=cnn



def constructNEPDF(geneTriple):
  # Define the 3 dimensions
  x = []
  y = []
  z = []
  # Extract the genes and label
  x_gene_name,y_gene_name,label = geneTriple[0],geneTriple[1],geneTriple[2]
  y.append(label)
  #z.append(x_gene_name+'\t'+y_gene_name)
  expressions = geneTriple[0]
  x_tf_bulk = [math.log10(float(x) + 10 ** -2) for x in list(expressions)]  ## 249 means the number of samples, users can just remove '[0:249]'
  expressionsY = geneTriple[1]
  x_gene_bulk = [math.log10(float(x) + 10 ** -2) for x in list(expressionsY)]
  H_T = numpy.histogram2d(x_tf_bulk, x_gene_bulk, bins=32)
  H= H_T[0].T
  #print(type(H))
  HT = (numpy.log10(H / 43261 + 10 ** -4) + 4)/4
  x.append(HT)
  xx = numpy.array(x)[:, :, :] 
  return (xx,y)

def predictGOslimCNN(gene):
  # Define the unknown gene
  unknownGene = gene
  # List to store the ones an zeros
  classVector = []
  for label in classes:
    # Load the CNN for this label
    cnn = labelCNNMap.get(label)
    if (type(cnn).__name__ == 'Sequential'):
      # Extract n (known) genes for each class
      #print('Extract genes for label: '+label)
      #noPairs = 10
      #positiveExamples = df[df[label]==1][0:noPairs][attributes]
      # TODO change to more pairs
      positiveExamples = df[df[label]==1][0:1][attributes]
      # Generate n Histograms
      # (gene1, gene2, label)
      # Iterate over every positive example
      currentPredictions = []
      for posGene in positiveExamples.values:
          # Construct the triple
          geneTriple = (posGene, unknownGene, '0')
          # Construct the NEPDF for this Gene pair
          geneHistogram = constructNEPDF(geneTriple)
          # Predict using the cnn
          predictionCNN = cnn.predict(numpy.expand_dims(numpy.array(geneHistogram[0]), axis=-1))
          currentPredictions.append(predictionCNN)
      classVector.append(numpy.mean(currentPredictions))
  return classVector




# Take the annotated non protein-coding genes into account
print('Start the function prediction for the genes of interest') 
predictionString = ''
targetX = genes_of_interest[attributes]
#targetY = annoNoPC[newClasses]

# Predict with each chain approach
# knnChain

knnChainTargetPred = chain_knn.predict_proba(targetX).toarray()
#print(clfChainTargetPred)
#knnChainTargetAUC = calculateAUC(knnChainTargetPred.toarray(), targetY, newClasses, type(chain_knn).__name__+'_knn_Target')
print('Predictions with '+type(chain_knn).__name__+'_knn done')

#print('Bug in predicting target date with '+type(chain_knn).__name__+'_knn')
#knnChainTargetAUC = ''



# rfChain
rfChainTargetPred = chain_rf.predict_proba(targetX)
#print(clfChainTargetPred)
#rfChainTargetAUC = calculateAUC(rfChainTargetPred.toarray(), targetY, newClasses, type(chain_rf).__name__+'_rf_Target')
print('Predictions with '+type(chain_rf).__name__+'_rf done')

#print('Bug in predicting target date with '+type(chain_rf).__name__+'_rf')
#rfChainTargetAUC = ''


print('Chain-SL-kNN: '+str(knnChainTargetAUC))
print('Chain-SL-RandomForest: '+str(rfChainTargetAUC))



# CNN-based approach (stored in labelCNNMap)
# Iterate ove every annotated non pc-gene and call predictGOslimCNN
cnnPredictions = []
for index, row in genes_of_interest.iterrows():
  geneGOslims = predictGOslimCNN(row[attributes])
  cnnPredictions.append(geneGOslims)
  #print('Gene '+str(index)+' classified')
predictionsString = 'geneID;'
for goslim in classes:
    predictionsString += str(goslim)+';'
predictionsString += '\n'
counter = 0
for index, row in genes_of_interest.iterrows():
    predictionsString += str(row[primaryKey]) + ';'
    cnnPredictionsString = numpy.array(cnnPredictions[counter])*100.00   
    for i in range(0, len(classes)):
        predictionsString += str(cnnPredictionsString[i]) + ';'
    
    counter += 1

    predictionsString += '\n'

with open("Predictions/PredictionsGenesOfInterest_Chain-SL-CNN.txt", "w") as prediction_file_cnn:
      prediction_file_cnn.write(predictionsString)











