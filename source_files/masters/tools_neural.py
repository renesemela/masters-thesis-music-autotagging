"""
========================= !!! READ ME !!! =========================
This script contains definitons of functions for work with OS.
Make sure you have installed all requirements from requirements.txt
===================================================================
"""

# Libraries: Import global
import numpy as np
import math
import h5py
import sqlite3
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.utils import Sequence
from keras.callbacks import Callback
from keras.models import Sequential, Model, load_model
from keras.layers import GRU, ELU, ZeroPadding2D, Reshape, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

# Libraries: Import custom
from masters.tools_system import makedir, currdate

# Matplotlib default parameters
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.rcParams['pdf.fonttype'] = 42

# Function: Make folder structure for results of new neural network
def folderstruct_network(network_name, dataset_name):
    makedir('./networks')
    dir_work = './networks/' + network_name + '_' + dataset_name + '_' + currdate() +  '/'
    makedir(dir_work)
    makedir(dir_work + 'model')
    makedir(dir_work + 'dataset')
    makedir(dir_work + 'results')
    makedir(dir_work + 'test')
    return dir_work

# Function: Definition of data generator for feeding neural network during training
class datagen(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        return np.array([np.load(file_name) for file_name in batch_x]), np.array(batch_y)

# Function: Definition of custom callback 
class model_custom_callback(Callback):
    def __init__(self, gen_val, gen_test, path_model, path_test_results, path_results, path_dataset_db, verbose=1):
        self.gen_val = gen_val
        self.gen_test = gen_test
        self.path_test_results = path_test_results
        self.path_results = path_results
        self.path_model = path_model
        self.path_dataset_db = path_dataset_db
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.acc = [None] 
        self.loss = [None]
        self.val_acc = [None] 
        self.val_loss = [None]
        self.val_auc = [None]
        self.test_auc = [None]
        self.best_epoch = 1

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.val_loss.append(logs.get('val_loss'))

        # Compute validation ROC-AUC
        print('\nComputing validation ROC-AUC: ')
        y_predicted = self.model.predict_generator(self.gen_val, verbose=self.verbose)
        y_true = self.gen_val[0][1]
        for i in range(1, len(self.gen_val)):
            y_true = np.concatenate([y_true, self.gen_val[i][1]])
        val_auc = roc_auc_score(y_true, y_predicted)
        self.val_auc.append(val_auc)
        print('val_auc: ' + str(round(val_auc,4)))

        # Compute test ROC-AUC
        print('\nComputing test ROC-AUC: ')
        y_predicted = self.model.predict_generator(self.gen_test, verbose=self.verbose)
        y_true = self.gen_test[0][1]
        for i in range(1, len(self.gen_test)):
            y_true = np.concatenate([y_true, self.gen_test[i][1]])
        test_auc = roc_auc_score(y_true, y_predicted)
        self.test_auc.append(test_auc)
        print('test_auc: ' + str(round(test_auc,4)))

        # Save all test predictions to file for future use
        epoch_str = str(epoch + 1)
        if len(epoch_str) == 1:
            epoch_str = '00' + epoch_str
        elif len(epoch_str) == 2:
            epoch_str = '0' + epoch_str
        np.save(self.path_test_results + 'model_test_predictions_ep' + epoch_str, y_predicted)

        # Save all metrics to model history file for future plotting
        np.savez(self.path_results + 'model_history.npz', 
            loss=self.loss, 
            acc=self.acc, 
            val_loss=self.val_loss, 
            val_acc=self.val_acc, 
            val_auc=self.val_auc, 
            test_auc=self.test_auc)

        # Print metrics summary for current epoch
        print('\n============ EPOCH ' + str(epoch+1) + ' SUMMARY ============')
        print('train_loss: \t' + str(round(self.loss[epoch+1], 4)))
        print('train_acc: \t' + str(round(self.acc[epoch+1], 4)))
        print('val_loss: \t' + str(round(self.val_loss[epoch+1], 4)))
        print('val_acc: \t' + str(round(self.val_acc[epoch+1], 4)))
        print('val_auc: \t' + str(round(self.val_auc[epoch+1], 4)))
        print('test_auc: \t' + str(round(self.test_auc[epoch+1], 4)))

        # Print overall summary of ROC-AUC values
        print('\n============ OVERALL SUMMARY ============')
        index_val_auc_max = np.argmax(np.array(self.val_auc[1:])) + 1
        val_auc_max = round(self.val_auc[index_val_auc_max], 4)
        index_test_auc_max = np.argmax(np.array(self.test_auc[1:])) + 1
        test_auc_max = round(self.test_auc[index_test_auc_max], 4)
        print('best val_auc: \t' + str(val_auc_max) + ' \t(epoch ' + str(index_val_auc_max) + ')')
        print('best test_auc: \t' + str(test_auc_max) + ' \t(epoch ' + str(index_test_auc_max) + ')')

        # Plot all relevant metrics and save plots to the results folder
        # Plot and save model loss (train, validation)
        plt.figure()
        plt.title('Model loss')
        plt.plot(self.loss)
        plt.plot(self.val_loss)
        plt.xlabel('epoch [-]')
        plt.ylabel(r'loss $\it{LOSS}$ [-]')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.grid(linewidth=0.25)
        plt.savefig(self.path_results + 'model_loss_train_validation.png')
        plt.savefig(self.path_results + 'model_loss_train_validation.pdf')
        plt.close()

        # Plot and save model accuracy (train, validation)
        plt.figure()
        plt.title('Model accuracy')
        plt.plot(self.acc)
        plt.plot(self.val_acc)
        plt.xlabel('epoch [-]')
        plt.ylabel(r'accuracy $\it{ACC}$ [-]')
        plt.legend(['train', 'validation'], loc='lower right')
        plt.grid(linewidth=0.25)
        plt.savefig(self.path_results + 'model_accuracy_train_validation.png')
        plt.savefig(self.path_results + 'model_accuracy_train_validation.pdf')
        plt.close()

        # Plot and save model ROC-AUC (validation)
        plt.figure()
        plt.title('Model ROC-AUC (validation data)')
        plt.plot(self.val_auc)
        plt.xlabel('epoch [-]')
        plt.ylabel(r'area under the ROC curve $\it{ROC}$-$\it{AUC}$ [-]')
        plt.grid(linewidth=0.25)
        plt.savefig(self.path_results + 'model_roc_auc_validation.png')
        plt.savefig(self.path_results + 'model_roc_auc_validation.pdf')
        plt.close()

        # Plot and save model ROC-AUC (test)
        plt.figure()
        plt.title('Model ROC-AUC (test data)')
        plt.plot(self.test_auc)
        plt.xlabel('epoch [-]')
        plt.ylabel(r'area under the ROC curve $\it{ROC}$-$\it{AUC}$ [-]')
        plt.grid(linewidth=0.25)
        plt.savefig(self.path_results + 'model_roc_auc_test.png')
        plt.savefig(self.path_results + 'model_roc_auc_test.pdf')
        plt.close()

        print('\nTemporary results are also located at: ' + self.path_results)

        # Custom Early Stopping (stop when test ROC-AUC is not improving)
        patience = 20
        if val_auc > self.val_auc[self.best_epoch]:
            self.best_epoch = epoch+1
        print('\nEarly stopping patience: ' + str((epoch+1)-self.best_epoch) + '/' + str(patience) + ' epochs')
        if epoch+1 >= self.best_epoch+patience:
            if self.verbose == 1:
                print('Epoch ' + str(epoch+1) + ': Early stopping.')
            self.model.stop_training = True
        print('\n===================================================================================================================================================\n')
    
    def on_train_end(self, epoch):
        # Find epoch with the best test ROC-AUC
        index = np.argmax(np.array(self.test_auc[1:])) + 1 # First element in self.test_auc is None. Index + 1 is used because of indexing shift caused by that.
        print('Highest test ROC-AUC: ' + str(round(self.test_auc[index], 4)) + ' (epoch ' + str(index) + ')')
        
        # Load the best model and predict on test data
        print('\nComputing ROC curves for the best epoch: ')
        filenames_model = np.sort([i for i in os.listdir(self.path_model) if os.path.isfile(os.path.join(self.path_model,i)) and 'model_checkpoint_' in i])
        model = load_model(self.path_model + filenames_model[index-1]) # [index-1] because list of files is indexed from 0 not from 1
        y_predicted = model.predict_generator(self.gen_test, verbose=self.verbose)
        y_true = self.gen_test[0][1]
        for i in range(1, len(self.gen_test)):
            y_true = np.concatenate([y_true, self.gen_test[i][1]])
        # Save the best model to results folder
        model.save(self.path_results + 'bestepoch_ep' + str(index) + '_model.hdf5')

        # Plot and save ROC curves for the best epoch
        # Prepare variables for ROC curves
        n_classes = np.size(y_true,1)
        fpr = dict()
        tpr = dict()
        tresh = dict()
        roc_auc = dict()

        # Compute fpr, tpr, roc_auc and do macro average
        for i in range(n_classes):
            fpr[i], tpr[i], tresh[i]  = roc_curve(y_true[:,i], y_predicted[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

        # Plot and save macro average ROC curve of all classes
        plt.figure()
        plt.title('ROC curve (test data, epoch ' + str(index) + ')')
        plt.plot(fpr['macro'], tpr['macro'], label='macro average (AUC = ' + str(round(roc_auc['macro'], 4)) + ')')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel(r'False Positive Rate $\it{FPR}$ [-]')
        plt.ylabel(r'True Positive Rate $\it{TPR}$ [-]')
        plt.legend(loc='lower right')
        plt.grid(linewidth=0.25)
        plt.savefig(self.path_results + 'bestepoch_ep' + str(index) + '_roc_average.png')
        plt.savefig(self.path_results + 'bestepoch_ep' + str(index) + '_roc_average.pdf')
        plt.close()

        # Connect to the dataset database file and get tags names for nicer plotting
        conn = sqlite3.connect(self.path_dataset_db)
        c = conn.cursor()
        c.execute('SELECT * FROM tags')
        names_classes = [description[0] for description in c.description]
        conn.close()

        # Plot and save ROC curves of all classes into one figure
        plt.figure(figsize=(13.2,8.4))
        ax = plt.subplot(111)
        ax.set_title('ROC curves of all classes (test data, epoch ' + str(index) + ')')
        ax.plot(fpr['macro'], tpr['macro'], label='macro average (AUC = ' + str(round(roc_auc['macro'], 2)) + ')', color='black', linestyle=':', linewidth=4)
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=str(names_classes[i]) + ' (AUC = ' + str(round(roc_auc[i], 2)) + ')', linewidth=1)
        ax.plot([0, 1], [0, 1], 'k--')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 1.3])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel(r'False Positive Rate $\it{FPR}$ [-]')
        ax.set_ylabel(r'True Positive Rate $\it{TPR}$ [-]')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize='small', labelspacing=0.1)
        ax.grid(linewidth=0.25)
        plt.savefig(self.path_results + 'bestepoch_ep' + str(index) + '_roc_classes.png', bbox_inches='tight')
        plt.savefig(self.path_results + 'bestepoch_ep' + str(index) + '_roc_classes.pdf', bbox_inches='tight')
        plt.close()

        print('\n                      ██████╗  ██████╗ ███╗   ██╗███████╗                      ')
        print('                      ██╔══██╗██╔═══██╗████╗  ██║██╔════╝                      ')
        print('█████╗█████╗█████╗    ██║  ██║██║   ██║██╔██╗ ██║█████╗      █████╗█████╗█████╗')
        print('╚════╝╚════╝╚════╝    ██║  ██║██║   ██║██║╚██╗██║██╔══╝      ╚════╝╚════╝╚════╝')
        print('                      ██████╔╝╚██████╔╝██║ ╚████║███████╗                      ')
        print('                      ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚══════╝                      ')
        print('\nDone! Results are located at ' + self.path_results + '\n')

# Function: Definition of neural networks compiler
def model_compiler(model):
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['acc'])

# Function: Definition of FCNN3 layers
def model_build_fcnn3(n_classes):
    input_shape = (96,1366)
    
    model = Sequential(name='fcnn3')

    model.add(Reshape(target_shape=input_shape + (1,), input_shape=input_shape))
    model.add(ZeroPadding2D(padding=(0, 37)))

    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4,8)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4,10)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(6,18)))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=n_classes, activation='sigmoid'))

    model_compiler(model)

    return model

# Function: Definition of FCNN4 layers
def model_build_fcnn4(n_classes):
    input_shape = (96,1366)
    
    model = Sequential(name='fcnn4')

    model.add(Reshape(target_shape=input_shape + (1,), input_shape=input_shape))
    model.add(ZeroPadding2D(padding=(0, 37)))

    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,5)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(3,6)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4,6)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4,8)))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=n_classes, activation='sigmoid'))

    model_compiler(model)

    return model

# Function: Definition of FCNN5 layers
def model_build_fcnn5(n_classes):
    input_shape = (96,1366)
    
    model = Sequential(name='fcnn5')

    model.add(Reshape(target_shape=input_shape + (1,), input_shape=input_shape))
    model.add(ZeroPadding2D(padding=(0, 37)))

    model.add(BatchNormalization(axis=3))
    
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,4)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,5)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(3,6)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4,6)))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=n_classes, activation='sigmoid'))

    model_compiler(model)

    return model

# Function: Definition of FCNN6 layers
def model_build_fcnn6(n_classes):
    input_shape = (96,1366)
    
    model = Sequential(name='fcnn6')

    model.add(Reshape(target_shape=input_shape + (1,), input_shape=input_shape))
    model.add(ZeroPadding2D(padding=(0, 37)))

    model.add(BatchNormalization(axis=3))
    
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,3)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,4)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,5)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(3,6)))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=n_classes, activation='sigmoid'))

    model_compiler(model)

    return model

# Function: Definition of CRNN3 layers
def model_build_crnn3(n_classes):
    input_shape = (96,1366)
    
    model = Sequential(name='crnn3')

    model.add(Reshape(target_shape=input_shape + (1,), input_shape=input_shape))
    model.add(ZeroPadding2D(padding=(0, 37)))

    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4,8)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4,10)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(6,18)))
    model.add(Dropout(rate=0.25))

    model.add(Reshape(target_shape=(1,128)))
    model.add(GRU(units=256, return_sequences=True))
    model.add(GRU(units=256, return_sequences=False))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=n_classes, activation='sigmoid'))

    model_compiler(model)

    return model

# Function: Definition of CRNN4 layers
def model_build_crnn4(n_classes):
    input_shape = (96,1366)
    
    model = Sequential(name='crnn4')

    model.add(Reshape(target_shape=input_shape + (1,), input_shape=input_shape))
    model.add(ZeroPadding2D(padding=(0, 37)))

    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,5)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(3,6)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4,6)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4,8)))
    model.add(Dropout(rate=0.25))

    model.add(Reshape(target_shape=(1,256)))
    model.add(GRU(units=256, return_sequences=True))
    model.add(GRU(units=256, return_sequences=False))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=n_classes, activation='sigmoid'))

    model_compiler(model)

    return model

# Function: Definition of CRNN5 layers
def model_build_crnn5(n_classes):
    input_shape = (96,1366)
    
    model = Sequential(name='crnn5')

    model.add(Reshape(target_shape=input_shape + (1,), input_shape=input_shape))
    model.add(ZeroPadding2D(padding=(0, 37)))

    model.add(BatchNormalization(axis=3))
    
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,4)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,5)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(3,6)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4,6)))
    model.add(Dropout(rate=0.25))

    model.add(Reshape(target_shape=(1,512)))
    model.add(GRU(units=256, return_sequences=True))
    model.add(GRU(units=256, return_sequences=False))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=n_classes, activation='sigmoid'))

    model_compiler(model)

    return model

# Function: Definition of CRNN6 layers
def model_build_crnn6(n_classes):
    input_shape = (96,1366)
    
    model = Sequential(name='crnn6')

    model.add(Reshape(target_shape=input_shape + (1,), input_shape=input_shape))
    model.add(ZeroPadding2D(padding=(0, 37)))

    model.add(BatchNormalization(axis=3))
    
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,3)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,4)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,5)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(3,6)))
    model.add(Dropout(rate=0.25))

    model.add(Reshape(target_shape=(1,1024)))
    model.add(GRU(units=256, return_sequences=True))
    model.add(GRU(units=256, return_sequences=False))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=n_classes, activation='sigmoid'))

    model_compiler(model)

    return model