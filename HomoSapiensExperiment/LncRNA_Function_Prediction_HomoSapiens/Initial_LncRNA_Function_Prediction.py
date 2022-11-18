# Imports 
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
      pip.main(['install', package])


#import_or_install('sklearn')
#import_or_install('numpy')
#import_or_install('pandas')
#import_or_install('tensorflow')
#!pip install scikit-multilearn
#!pip install libtlda
#!pip install --upgrade tables

import warnings
warnings.filterwarnings('ignore')
# os
import os
from os import path
# Data Preparation
import pandas as pd
import numpy
# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Preprocessing 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer

# Neural Network
from numpy import std
from sklearn.model_selection import RepeatedKFold

# sklearn classifier models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# save models
import pickle
import time

# sklearn Algorithm Adaption
from skmultilearn.adapt import MLkNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# sklearn Problem Transformation
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# plot
import matplotlib
import matplotlib.pyplot as plt

# sklearn metrics
from sklearn import metrics
import statistics

# model selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import h5py
import scipy.sparse as sparse

import os
import time
import random
os.environ['KERAS_BACKEND'] = 'theano'
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras import Input
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
import h5py
import scipy.sparse as sparse
# Splitting
from skmultilearn.model_selection import IterativeStratification
import math

print('All needed packages imported')

class Initial_model_build_and_train():
  df, dfNoPC, annoNoPC = None, None, None
  classes, attributes = [], []

  labelCNNMap = {}
  rf_chain = None
  knn_chain = None
  mlp, mlknn, rf = None, None, None


  trainX, trainY, testX, testY = None, None, None, None
  species = 'HomoSapiens'
  overrideClassifier = True
  overrideCNNModel = True
  expr_analysis = False
  primaryKey = 'geneId'

  def __init__(self, df_path, dfNoPC_path, annoNoPC_path ,dfShuffled_path):
    self.df_path = df_path
    self.dfNoPC_path = dfNoPC_path
    self.annoNoPC_path = annoNoPC_path

    # Create folder for the export of the performances and models
    self.createFolder('./Evaluation/ML_Approaches/')  
    self.createFolder('./Evaluation/Chain_Approaches/')  
    self.createFolder('./Evaluation/Chain_Approaches/CNN/')

    self.createFolder('./Models/ML_Approaches/')  
    self.createFolder('./Models/Chain_Approaches/')  
    self.createFolder('./Models/Chain_Approaches/CNN/')   

  def createFolder(self, directory):
      try:
        if not os.path.exists(directory):
          os.makedirs(directory)
      except OSError:
          print ('Error: Creating directory. ' +  directory) 
  
  def read_in_datasets(self):
    # Dataset - protein coding genes to learn with
    self.df = pd.read_csv(self.df_path) 
    # Dataset - lncRNA samples
    self.dfNoPC = pd.read_csv(self.dfNoPC_path)
    # Dataset - annotated non-protein coding genes to test the models
    self.annoNoPC = pd.read_csv(self.annoNoPC_path)
    # Shuffled set in order to get a baseline
    #self.dfShuffled = pd.read_csv(self.dfShuffled_path)

    # Read in the class and attribute names
    with open('ML_Datasets/classes.txt') as classesFile:
        classes = classesFile.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    self.classes = [x.strip() for x in classes] 

    # Read in the class and attribute names
    with open('ML_Datasets/attributes.txt') as attributesFile:
        attributes = attributesFile.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    self.attributes = [x.strip() for x in attributes] 

  def iterative_train_test_split(self, X, y, test_size):
    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0-test_size])
    train_indexes, test_indexes = next(stratifier.split(X, y))
    X_train, y_train = X.loc[train_indexes], y.loc[train_indexes, :]
    X_test, y_test = X.loc[test_indexes], y.loc[test_indexes, :]
    print('X_train: '+str(X_train.shape))
    print('X_test: '+str(X_test.shape))
    return X_train, y_train, X_test, y_test

  def split_datasets(self):
    # Split the dataset
    print('Split the dataset')
    X=self.df[self.attributes].reset_index(drop=True)
    y=self.df[self.classes].reset_index(drop=True)
    X_train, y_train, X_test, y_test = self.iterative_train_test_split(X, y, test_size = 0.1)
    X=None
    y=None
    
    # ML-SMOTE excluded

    # Extract the labels and expressions
    self.trainY = y_train[self.classes]
    self.trainX = X_train[self.attributes]
    self.testY = y_test[self.classes]
    self.testX = X_test[self.attributes]

    # ML-SMOTE excluded

    print('Train and Test set imported with '+str(self.trainY.shape[1])+' classes and '+str(self.trainX.shape[1])+' features')


  def calculateAUC(self, predictionAUC, testYAUC, classes, modelName):

    # Define the evaluation path
    if ('binaryrelevance' in modelName.lower()):
      evaluationPath = 'Evaluation/Chain_Approaches/'
    elif ('convolutional' in modelName.lower()):
      evaluationPath = 'Evaluation/Chain_Approaches/CNN/'
    else:
      evaluationPath = 'Evaluation/ML_Approaches/'

    # Calculate fpr and tpr for every GOslim
    aucs = {}
    prs = {}
    for j in range(0, len(classes)):
        testYSingle = [i[j] for i in testYAUC[classes].values]
        yhatSingle = [k[j] for k in predictionAUC]
        precision, recall, thresholds = metrics.precision_recall_curve(testYSingle, yhatSingle)
        aucs[classes[j]] = metrics.auc(recall, precision)
        prs[classes[j]] = [precision, recall]
    # Build a dataframe consisting of all AUCs
    #aucsDf = pd.DataFrame(aucs.items(), columns=['class', 'AUC'], index=None).sort_values(by=['AUC'],ascending=False)
    aucsDf = pd.DataFrame(aucs.items(), columns=['class', 'AUC'], index=None)
    #aucsDf[['class', 'AUC']].sort_values(by='AUC',ascending=False).plot(kind='bar')
    x = list(aucsDf['class'])
    y = list(aucsDf['AUC'])
    heights = aucsDf['AUC']
    bars = aucsDf['class']
    y_pos = range(len(bars))
    fig = plt.figure(figsize=(15,5))
    fig.suptitle('AUC Distribution for the '+modelName+' Model', fontsize=20)
    plt.xlabel('GOslim Annotation', fontsize=18)
    plt.ylabel('AUC', fontsize=16)
    plt.bar(y_pos, heights)
    # Rotation of the bars names
    plt.xticks(y_pos, bars, rotation=90, fontsize=14)
    plt.yticks(fontsize=14)
    #plt.gcf().subplots_adjust(bottom=0.4)
    plt.tight_layout()    
    plt.savefig(evaluationPath + modelName + '-Evaluation'+self.species+'.png',dpi=300, bbox_inches = "tight")
    # Export the Evaluation
    aucsDf.to_csv (evaluationPath + modelName + '-EvaluationDf'+self.species+'.csv', index = False, header=True)
    return aucsDf
  
  def extractGenes(self, goSlim, noPairs):
    # Extract the ensembl Ids and export for gene pair - generation
    print('Extract genes for label: '+goSlim)
    positiveExamples = list(self.df[self.df[goSlim]==1][0:noPairs][self.primaryKey])
    negativeExamples = list(self.df[self.df[goSlim]==0][0:noPairs][self.primaryKey])
    with open('GenePairs/genes_pos.txt', 'w') as f:
        for item in positiveExamples:
            f.write("%s\n" % item)
    with open('GenePairs/genes_neg.txt', 'w') as f:
        for item in negativeExamples:
            f.write("%s\n" % item)
  
    # Extract the gene expressions of the gene pairs and export as h5py-file
    genePairsDf = self.df[self.df[self.primaryKey].isin(list(positiveExamples + negativeExamples))][list(self.attributes)+[self.primaryKey]]
    # Create a h5 File
    #geneFile = h5py.File('mmusculusCNNCNew.h5','w')
    #geneFile.close()
    # Create a h5 Group
    geneIdsByte = [el.encode('UTF-8') for el in self.df[self.primaryKey].str.lower().to_numpy()]
    h5df = pd.DataFrame(pd.DataFrame({'axis0': geneIdsByte, 'axis1': list(range(1, len(self.df[self.primaryKey].str.lower())+1)), 'block0_items': geneIdsByte}))
    arr = sparse.csr_matrix(self.df[self.attributes].to_numpy())
    h5df['block0_values'] = arr.toarray().tolist()
    # Export the h5py gene expressions file
    h5df[['axis0','axis1','block0_items','block0_values']].to_hdf(self.species+'CNN.h5', key='rpkm', mode='w')
    # Check for correctness
    #b = h5py.File('mmusculusCNN.h5','r')
    #for i in b['rpkm']:
    #    print(b['rpkm'][i])
    #b.close()

  def get_in_out_gene_pair_list (self, test_index_list,train_index_list,cell_cycle_list_list,no_cycle_sample_list) :
    '''
    Yuan, Ye and Bar-Joseph, Ziv: Deep learning for inferring gene relationships from single-cell expression data
    '''
    test_cycle_gene = [cell_cycle_list_list[i] for i in test_index_list]
    test_no_cycle_gene = [no_cycle_sample_list[i] for i in test_index_list]
    train_cycle_gene = [cell_cycle_list_list[i] for i in train_index_list]
    train_no_cycle_gene = [no_cycle_sample_list[i] for i in train_index_list]
    cycle_cycle_test = []
    cycle_random_test = []
    cycle_cycle_train = []
    cycle_random_train = []
    # Generate gene pairs for training set
    for i in range (len(train_cycle_gene)):
        for j in range (len(test_index_list)):
            cycle_cycle_test.append(train_cycle_gene[i]+'\t'+test_cycle_gene[j]+'\t1')
            cycle_random_test.append(train_cycle_gene[i] + '\t' + test_no_cycle_gene[j]+'\t0')
    # Generate gene pairs for test set
    for i in range (len(train_index_list)):
        for j in range (len(train_index_list)):
            cycle_cycle_train.append(train_cycle_gene[i]+'\t'+train_cycle_gene[j]+'\t1')
            cycle_random_train.append(train_cycle_gene[i] + '\t' + train_no_cycle_gene[j]+'\t0')
    return (cycle_cycle_train+cycle_random_train,cycle_cycle_test+cycle_random_test)

  def generateGenePairs(self, goSlim):
    '''
    Yuan, Ye and Bar-Joseph, Ziv: Deep learning for inferring gene relationships from single-cell expression data
    '''
    print('Generate the gene pairs for label: '+goSlim)

    # Folder for the Histograms
    save_dir = os.path.join(os.getcwd(),'NEPDF_data')

    # Read in the positive gene examples
    positiveGenes = []
    s = open ('GenePairs/genes_pos.txt') ##function gene set downloaded for GSEA
    print(s)
    for line in s:
        gene = line.strip().lower()
        positiveGenes.append(gene)
        
    s.close()

    # Read in the negative gene examples
    negativeGenes = []
    s = open ('GenePairs/genes_neg.txt') ##function gene set downloaded for GSEA
    for line in s:
        gene = line.strip().lower()
        negativeGenes.append(gene)

    s.close()
    # Setup for the random gene pair genration
    random.seed(1)
    no_cycle_sample_list = random.sample(negativeGenes,len(negativeGenes))
    gene_list = [i for i in range(len(positiveGenes))]

    # Split into training and testing and call function to generate the pairs
    test_index_list = [i for i in range (int(numpy.ceil((1-1)*0.333*len(positiveGenes))),int(numpy.ceil(1*0.333*len(positiveGenes))))]
    train_index_list = [ i for i in gene_list if i not in test_index_list]
    #print (test_index_list,train_index_list)
    train_set, test_set = self.get_in_out_gene_pair_list(test_index_list,train_index_list,positiveGenes,negativeGenes)

    # Export the generated gene pairs
    save_dir = os.path.join(os.getcwd(), 'samplesxx/' + str(1) + '_samples')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    s = open(save_dir + '/samples.txt', 'w')
    for xxxx in train_set + test_set:
        s.write(xxxx + '\n')
    s.close()
    nums = [0, len(train_set), len(train_set + test_set)]
    s = open(save_dir + '/samples_nums.txt', 'w')
    for xxxx in nums:
        s.write(str(xxxx) + '\n')
    s.close()

  def get_sepration_index (self, file_name):
    '''
    Yuan, Ye and Bar-Joseph, Ziv: Deep learning for inferring gene relationships from single-cell expression data
    '''
    index_list = []
    s = open(file_name, 'r')
    for line in s:
      index_list.append(int(line))
    return (numpy.array(index_list))

  def generateNEPDF(self):
    '''
    Yuan, Ye and Bar-Joseph, Ziv: Deep learning for inferring gene relationships from single-cell expression data
    '''
    # Create a folder for NEPDF data
    save_dir = os.path.join(os.getcwd(),'NEPDF_data')
    if not os.path.isdir(save_dir):                 
        os.makedirs(save_dir)
        
    # Read in the gene expressions
    with h5py.File(self.species+'CNN.h5', 'r') as f:    
        rpkm = f['rpkm']

    # Read in the gene expressions
    print('Read in gene expressions')
    store = pd.HDFStore(self.species+'CNN.h5')
    rpkm = store['rpkm']
    store.close()

    gene_pair_label = []
    s=open('samplesxx/1_samples/samples.txt')
    for line in s:
        gene_pair_label.append(line)
    gene_pair_index = self.get_sepration_index('samplesxx/1_samples/samples_nums.txt')#'mmukegg_new_new_unique_rand_labelx_num.npy')#sys.argv[6]) # read file speration index
    s.close()
    gene_pair_label_array = numpy.array(gene_pair_label)

    # Generate the histograms
    counter = 1
    for i in range(len(gene_pair_index)-1):   #### many sperations
        start_index = gene_pair_index[i]
        print(i)
        print(start_index)
        end_index = gene_pair_index[i+1]
        print(end_index)
        x = []
        y = []
        z = []
        
        for gene_pair in gene_pair_label_array[start_index:end_index]: ## each speration
            # Extract the pairs 
            separation = gene_pair.split()
            x_gene_name,y_gene_name,label = separation[0],separation[1],separation[2]
            y.append(label)
            z.append(x_gene_name+'\t'+y_gene_name)
            expressions = rpkm.loc[rpkm['block0_items'] == x_gene_name.encode('utf8').lower()]['block0_values']
            x_tf_bulk = [math.log10(float(x) + 10 ** -2) for x in list(expressions)[0]]  ## 249 means the number of samples, users can just remove '[0:249]'
            expressionsY = rpkm.loc[rpkm['block0_items'] == y_gene_name.encode('utf8').lower()]['block0_values']
            x_gene_bulk = [math.log10(float(x) + 10 ** -2) for x in list(expressionsY)[0]]
            H_T = numpy.histogram2d(x_tf_bulk, x_gene_bulk, bins=32)
            H= H_T[0].T
            #print(type(H))
            HT = (numpy.log10(H / 43261 + 10 ** -4) + 4)/4
            x.append(HT)
        xx = numpy.array(x)[:, :, :] 
        #print('----')
        #print(len(xx))
        counter+=1
        #print('----')
        numpy.save(save_dir+'/Nxdata_tf' + str(i) + '.npy', xx)
        numpy.save(save_dir+'/ydata_tf' + str(i) + '.npy', numpy.array(y))
        numpy.save(save_dir+'/zdata_tf' + str(i) + '.npy', numpy.array(z))
  
  def load_data_TF2(self, indel_list,data_path):
    '''
    Yuan, Ye and Bar-Joseph, Ziv: Deep learning for inferring gene relationships from single-cell expression data
    '''
    xxdata_list = []
    yydata = []
    count_set = [0]
    count_setx = 0
    for i in indel_list:#len(h_tf_sc)):
        xdata = numpy.load(data_path+'/Nxdata_tf' + str(i) + '.npy')
        ydata = numpy.load(data_path+'/ydata_tf' + str(i) + '.npy')
        for k in range(len(ydata)):
          xxdata_list.append(xdata[k,:,:])
          yydata.append(ydata[k])
        count_setx = count_setx + len(ydata)
        count_set.append(count_setx)
        print (i,len(ydata))
    yydata_array = numpy.array(yydata)
    yydata_x = yydata_array.astype('int')
    print (numpy.array(xxdata_list).shape)
    return ((numpy.array(xxdata_list),yydata_x,count_set))

  def generateAndTrainCNN(self, goslim, historyPath):
    '''
    Yuan, Ye and Bar-Joseph, Ziv: Deep learning for inferring gene relationships from single-cell expression data
    '''
    # Properties
    data_augmentation = False
    # num_predictions = 20
    batch_size = 512 # mini batch for training
    num_classes = 2   #### categories of labels
    epochs = 60    #### iterations of trainning, with GPU 1080, each epoch takes about 60s
    #length_TF =3057  # number of divide data parts
    # num_predictions = 20
    model_name = 'CNN_'+goslim+'.h5'
    test_indel = 0
    data_path = 'NEPDF_data' ### XX_new is the NEPDF path
    test_TF = [1]
    train_TF = [0]
    (x_train, y_train,count_set_train) = self.load_data_TF2(train_TF,data_path)
    (x_test, y_test,count_set) = self.load_data_TF2(test_TF,data_path)
    print(x_train.shape, 'x_train samples')
    print(x_test.shape, 'x_test samples')

    #save_dir = os.path.join(os.getcwd(),str(test_indel)+'_y_001_XXXXXXXXXXX_saved_models_T_32-32-64-64-128-128-512_e'+str(epochs))
    save_dir = os.path.join(os.getcwd(),'Models/Chain_Approaches/CNN')
    
    print(y_train.shape, 'y_train samples')
    print(y_test.shape, 'y_test samples')

    # Set the directory for the export
    if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    # Build the CNN
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(32,32,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3),activation='relu') )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3),padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3),padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3),padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))
    #flatten
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    adam = Adam(learning_rate=0.01) 

    model.compile(loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics='accuracy')

    # Manually shuffle the data
    a = x_train
    b = y_train
    c = list(zip(a, b))
    random.shuffle(c)
    x_trainShuff, y_trainShuff = zip(*c)

    print(numpy.array(x_trainShuff).shape)

    if not data_augmentation:
            print('Not using data augmentation.')
            history = model.fit(numpy.expand_dims(numpy.array(x_trainShuff), axis=-1), numpy.array(y_trainShuff),
                      batch_size=batch_size,
                      epochs=epochs,validation_split=0.1,    #change expochs to 200!
                      shuffle=True)
    # Save the model
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.
    #scores = model.evaluate(numpy.expand_dims(numpy.array(x_test), axis=-1), y_test, verbose=1)
    # summarize history for accuracy
    print(history.history.keys())
    # Visualize evaluation IDEA: save all historys to construct a big one for all CNNs
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy '+goslim)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Evaluation/Chain_Approaches/CNN/CNN-Accuracy-'+goslim+'-Evaluation.png',dpi=300, bbox_inches = "tight")

    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss '+goslim)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Evaluation/Chain_Approaches/CNN/CNN-Loss-'+goslim+'-Evaluation.png',dpi=300, bbox_inches = "tight")
    plt.show()
    # Save the history
    with open(str(historyPath)+goslim, 'wb') as file_pi:
          pickle.dump(history.history, file_pi)



  def train_and_build_CNN_Chain_approach(self):
    if not os.path.exists('GenePairs'):
      os.makedirs('GenePairs')
    # Construct a map label --> CNN
    labelCNNMap = {}
    historyPath = 'Evaluation/Chain_Approaches/CNN/trainHistoryDict-CNN-'
    print('Start training the Chain-SL-CNN model')
    # Do it for every class:
    for label in self.classes:
      try:
        # Try to load the existing classifier
        cnn = load_model('Models/CNN/CNN_'+label+'.h5')
      except:
        cnn = None
        pass
      if (cnn == None or self.overrideCNNModel == True):
        # Extract the genes and store the ensemb Ids and expressions
        print('Extract genes')
        self.extractGenes(label, 100)
        # Generate the gene pairs
        print('Generate gene pairs')
        self.generateGenePairs(label)
        # Generate the NEPDF data for the training
        print('Generating NEPDF histograms')
        self.generateNEPDF()
        # Build an train the CNNs
        print('Train CNN')
        self.generateAndTrainCNN(label, historyPath)
        print('new CNN_'+label+'.h5 ' +' saved')
        # Save in the map
        labelCNNMap[label]=load_model('Models/Chain_Approaches/CNN/CNN_'+label+'.h5')
        print('CNN model retrieved')
      else:
        print('CNN_'+label+'.h5 ' +'found and loaded')
        # Save in the map
        labelCNNMap[label]=cnn
      print('-----------------')
    
    # Save the CNN approach
    self.labelCNNMap = labelCNNMap

  def constructNEPDF(self, geneTriple):
    '''
    Yuan, Ye and Bar-Joseph, Ziv: Deep learning for inferring gene relationships from single-cell expression data
    '''
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
    
  def predictGOslimCNN(self, gene):
    # Define the unknown gene
    unknownGene = gene
    # List to store the ones an zeros
    classVector = []

    for label in self.classes:
      # Load the CNN for this label
      cnn = self.labelCNNMap.get(label)
      if (type(cnn).__name__ == 'Sequential'):
        # Extract n (known) genes for each class
        noPairs = 40
        positiveExamples = self.df[self.df[label]==1][0:noPairs][self.attributes]
        #positiveExamples = self.df[self.df[label]==1][self.attributes]
        # Generate n Histograms
        # (gene1, gene2, label)
        # Iterate over every positive example
        currentPredictions = []
        for posGene in positiveExamples.values:
            # Construct the triple
            geneTriple = (posGene, unknownGene, '0')
            # Construct the NEPDF for this Gene pair
            geneHistogram = self.constructNEPDF(geneTriple)
            # Predict using the cnn
            predictionCNN = cnn.predict(numpy.expand_dims(numpy.array(geneHistogram[0]), axis=-1))
            currentPredictions.append(predictionCNN)
        classVector.append(numpy.mean(currentPredictions))
    return classVector

  def evaluate_CNN_approach(self):
    # Iterate ove every annotated non pc-gene and call predictGOslimCNN
    cnnPredictionsValidation = []
    for index, row in self.testX.iterrows():
        geneGOslims = self.predictGOslimCNN(row[self.attributes])
        cnnPredictionsValidation.append(geneGOslims)
    # Evaluate the performance
    cnnPredictions = self.calculateAUC(cnnPredictionsValidation, self.testY, self.classes, 'ConvolutionalNN')

  def train_and_build_knn_chain(self, kneighbors):
    start = time.time()
    # Define the chain of kNN classifiers
    try:
      # Try to load the existing classifier
      knn_chain = pickle.load(open('Models/Chain_Approaches/Chain_knn_'+self.species+'.sav', 'rb'))
    except:
      knn_chain = None
      pass
    if (knn_chain == None or self.overrideClassifier == True):
      knn_chain = BinaryRelevance(classifier=KNeighborsClassifier(n_neighbors=kneighbors))
      print('Start training the Chain-SL-kNN model')
      knn_chain.fit(self.trainX, self.trainY)
      # Save the Chain-SL-kNN
      filename = 'Models/Chain_Approaches/Chain_knn_'+self.species+'.sav'
      pickle.dump(knn_chain, open(filename, 'wb'))
      print('New Chain_knn_'+self.species+' saved')
    else:
      print('loaded Chain_knn_'+self.species+' found')
      pass
    knn_chain_Prediction = knn_chain.predict(self.testX)
    #print(clfPrediction[0])
    self.knn_chain = knn_chain
    # Evaluate the performance
    self.calculateAUC(knn_chain_Prediction.toarray(), self.testY, self.classes, type(self.knn_chain).__name__+'-kNN')
    end = time.time()
    print(str(end-start)+' seconds to train and evaluate the Chain-SL-kNN')
    
  def train_and_build_rf_chain(self, maxdepth):
    start = time.time()
    # Define the chain of RandomForests
    try:
      # Try to load the existing classifier
      rf_chain = pickle.load(open('Models/Chain_Approaches/Chain_RandomForest_'+self.species+'.sav', 'rb'))
    except:
      rf_chain = None
      pass
    if (rf_chain == None or self.overrideClassifier == True):
      rf_chain = BinaryRelevance(classifier=RandomForestClassifier(max_depth=maxdepth, random_state=0))
      print('Start training the Chain-SL-RandomForest model')
      rf_chain.fit(self.trainX, self.trainY)
      # Save the Chain-SL-kNN
      filename = 'Models/Chain_Approaches/Chain_RandomForest_'+self.species+'.sav'
      pickle.dump(rf_chain, open(filename, 'wb'))
      print('New Chain_RandomForest_'+self.species+' saved')
    else:
      print('loaded Chain_RandomForest_'+self.species+' found')
      pass    
    rf_chain_Prediction = rf_chain.predict(self.testX)
    #print(clfPrediction[0])
    self.rf_chain = rf_chain
    # Evaluate the performance
    self.calculateAUC(rf_chain_Prediction.toarray(), self.testY, self.classes, type(self.rf_chain).__name__+'-RandomForest_Depth')
    end = time.time()
    print(str(end-start)+' seconds to train andevaluate the Chain-SL-RandomForest')


  def build_ML_neural_network(self, n_inputs, n_outputs):
    mlp = Sequential()
    mlp.add(Dense(n_outputs*8, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    mlp.add(BatchNormalization())
    mlp.add(LeakyReLU())
    mlp.add(Dropout(0.2))
    mlp.add(Dense(n_outputs*4, kernel_initializer='he_uniform', activation='relu'))
    mlp.add(BatchNormalization())
    mlp.add(LeakyReLU())
    mlp.add(Dropout(0.2))
    mlp.add(Dense(n_outputs*2, kernel_initializer='he_uniform', activation='relu'))
    mlp.add(BatchNormalization())
    mlp.add(LeakyReLU())
    mlp.add(Dropout(0.2))
    mlp.add(Dense(n_outputs, activation='sigmoid'))

    metric= AUC(
        num_thresholds=200, curve='PR',
        summation_method='interpolation', name=None, dtype=None,
        thresholds=None, multi_label=True, label_weights=None
    )
    mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return mlp


  def train_and_build_MLC_approaches(self):

    # Try to load if already trained MLP 
    try:
      mlp = load_model('Models/ML_Approaches/Sequential_'+self.species+'.h5')
    except:
      mlp = None
        
    if (mlp == None or self.overrideClassifier == True): 
      # Build a MLP 
      n_inputs, n_outputs = self.trainX.shape[1], self.trainY.shape[1]
      mlp = self.build_ML_neural_network(n_inputs, n_outputs)
      # Fit the MLP
      print('Start training the ML-MLP model')
      mlp.fit(self.trainX, self.trainY, epochs=100, batch_size=32, verbose=0)
      # Save the MLP
      mlp.save('Models/ML_Approaches/Sequential_'+self.species+'.h5')
      print('New Sequential_'+self.species+' saved')
    else:
      print('loaded Sequential_'+self.species+' found')
      pass
    # Predict
    predictionsMLP = mlp.predict(self.testX)
    # Evaluate the performance
    aucMLP = self.calculateAUC(predictionsMLP, self.testY, self.classes, type(mlp).__name__)
    print('Sequential_'+self.species+' evaluated')

    # ML-SMOTE excluded

    # Save the MLP
    self.mlp = mlp
    print('MLP training and evaluation finished')
    # ------------------------------------------------------------------------------------
    # MLknn approach

    # MLknn
    try:
      # Try to load the existing classifier
      mlknn = pickle.load(open('Models/ML_Approaches/MLknn_'+self.species+'.sav', 'rb'))
    except:
      mlknn = None
      pass
    if (mlknn == None or self.overrideClassifier == True): 
      # Construct a Multi Label kNN Model
      mlknn = MLkNN(k=20)
      # Train the MLknn
      print('Start training the ML-kNN model')
      mlknn.fit(self.trainX.to_numpy(), self.trainY.to_numpy())
      # Save MLknn
      filename = 'Models/ML_Approaches/MLknn_'+self.species+'.sav'
      pickle.dump(self.mlknn, open(filename, 'wb'))
      print('New MLknn saved')
    else:
      print('loaded MLknn found')
      pass

    # ML-SMOTE got excluded

    # predict
    predictionskNN = mlknn.predict_proba(self.testX.to_numpy()).toarray()
    # Evaluate the performance
    aucMLknn = self.calculateAUC(predictionskNN, self.testY, self.classes, type(mlknn).__name__)
    print('MLknn_'+self.species+' evaluated')

    # Save MLknn
    self.mlknn = mlknn

    #--------------------------------------------------------------------------------------------------------------------
    
    # RF
    try:
      # Try to load the existing classifier
      rf = pickle.load(open('Models/ML_Approaches/RandomForest_'+self.species+'.sav', 'rb'))
    except:
      rf = None
      pass
    if (rf == None or self.overrideClassifier == True): 
      # Construct a Random Forest
      rf = RandomForestClassifier(random_state=0)
      print('Start training the ML-RandomForest model')
      rf.fit(self.trainX, self.trainY.to_numpy())
      # Save the RandomForest
      filename = 'Models/ML_Approaches/RandomForest_'+self.species+'.sav'
      pickle.dump(rf, open(filename, 'wb'))
      print('New RandomForest_'+self.species+' saved')
    else:
      print('loaded RandomForest_'+self.species+' found')
      pass

    # predict
    predictionsRf = rf.predict_proba(self.testX)
    # Transpose the prediction matrix 
    predictionsRfTrans = list(map(list, zip(*predictionsRf)))
    # Discard the probability not being in a class
    for i in range(0,len(predictionsRfTrans)):
      for j in range(0,len(predictionsRfTrans[i])):
        predictionsRfTrans[i][j] = numpy.delete(predictionsRfTrans[i][j], 0)[0]

    # Evaluate the performance
    rfAUC = self.calculateAUC(predictionsRfTrans, self.testY, self.classes, type(rf).__name__)
    print('RandomForest_'+self.species+' evaluated')

    # ML-SMOTE got excluded

    """### Analyse the importance of features during training"""
    if (self.expr_analysis):
        print('Analyse the importance of features based on RandomForest')
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.set_title('Visualization - Feature importances '+self.species)
        # Extract the importances out of the model
        feat_importances = pd.Series(rf.feature_importances_, index=self.attributes)
        feat_importances.nlargest(20).plot(kind='barh',label='Feature importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.legend(loc='best')
        plt.savefig('Evaluation/'+self.species+'FeatureImportance.png',dpi=300, bbox_inches = "tight")

    # Save RF
    self.rf = rf

    #--------------------------------------------------------------------------------------------------
    # Construct the dataframe consisting of all evaluations of the models
    dfEvalPlot = None
    dfEvalPlot = pd.merge(aucMLP, aucMLknn, on='class')
    dfEvalPlot = pd.merge(dfEvalPlot, rfAUC, on='class')
    dfEvalPlot.columns = ['class','mlpAUC','mlknnAUC','rfAUC']
    #df.loc[df['column_name'] == some_value]

    # Plot the comparison of the evaluations
    dfEvalPlot.loc[dfEvalPlot['class'].isin(aucMLP['class'])].plot(align='center',kind='bar', rot=90, width=0.8, figsize=(35,10), legend=True, fontsize=18, title='AUC (PR) distribution for all models')
    plt.xticks(range(len(aucMLP['class'])), aucMLP['class'], rotation='vertical')
    plt.tight_layout()
    plt.xlabel('GOslim Annotation', fontsize=35)
    plt.ylabel('AUC (Precision recall)', fontsize=35)
    plt.legend(loc=1, prop={'size': 18})

    plt.savefig('Evaluation/EvaluationComparisonAlgorithmAdaptation.png',dpi=300)

# Instantiate a initial build and train object
initial_build_and_train = Initial_model_build_and_train('ML_Datasets/PCgenes.csv', 'ML_Datasets/lncRNAs.csv', 'ML_Datasets/annoNPCgenes.csv', 'ML_Datasets/shuffledGenes.csv')
# Read in the datasets
initial_build_and_train.read_in_datasets()
# Initial split
initial_build_and_train.split_datasets()




#---------------------------------------------------------------------------
# CHAIN BASED MODELS
#---------------------------------------------------------------------------

# Train and evaluate the Chain-SL-kNN model
initial_build_and_train.train_and_build_knn_chain(20)
# Train and evaluate the Chain-SL-RandomForest model
initial_build_and_train.train_and_build_rf_chain(3)

# Start building and training of the Chain-SL-CNN model
initial_build_and_train.train_and_build_CNN_Chain_approach()
# Evaluate the Chain-SL-CNN - model
initial_build_and_train.evaluate_CNN_approach()

#--------------------------------------------------------------------------
# ALGORITHM ADAPTATION MODELS
#--------------------------------------------------------------------------

# Train and evaluate all three adapted models (ML-MLP, ML-kNN, ML-RandomForest)
initial_build_and_train.train_and_build_MLC_approaches()


