import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from itertools import product
from typing import Dict
from ax.service.managed_loop import optimize
from ax.plot.contour import plot_contour
from ax.utils.notebook.plotting import render
import copy

"""
@copyright for dataset:

DATA SET DESCRIPTION 
The Avila data set has been extracted from 800 images of the the "Avila Bible", 
a giant Latin copy of the whole Bible produced during the XII century between Italy and Spain.  
The palaeographic analysis of the  manuscript has  individuated the presence of 12 copyists. 
The pages written by each copyist are not equally numerous. 
Each pattern contains 10 features and corresponds to a group of 4 consecutive rows.

The prediction task consists in associating each pattern to one of the 12 copyists 
(labeled as: A, B, C, D, E, F, G, H, I, W, X, Y).
The data have has been normalized, by using the Z-normalization method.

Class distribution (training set)
A: 4286
B: 5  
C: 103 
D: 352 
E: 1095 
F: 1961 
G: 446 
H: 519
I: 831
W: 44
X: 522 
Y: 266

ATTRIBUTE DESCRIPTION

ID      Name    
F1       intercolumnar distance 
F2       upper margin 
F3       lower margin 
F4       exploitation 
F5       row number 
F6       modular ratio 
F7       interlinear spacing 
F8       weight 
F9       peak number 
F10     modular ratio/ interlinear spacing
Class: A, B, C, D, E, F, G, H, I, W, X, Y


CITATIONS
If you want to refer to the Avila data set in a publication, please cite the following paper:

C. De Stefano, M. Maniaci, F. Fontanella, A. Scotto di Freca,
Reliable writer identification in medieval manuscripts through page layout features: 
The "Avila" Bible case, Engineering Applications of Artificial Intelligence, Volume 72, 2018, pp. 99-110.
"""

"""
@copyright for sklearn API:
@inproceedings{sklearn_api,
  author    = {Lars Buitinck and Gilles Louppe and Mathieu Blondel and
               Fabian Pedregosa and Andreas Mueller and Olivier Grisel and
               Vlad Niculae and Peter Prettenhofer and Alexandre Gramfort
               and Jaques Grobler and Robert Layton and Jake VanderPlas and
               Arnaud Joly and Brian Holt and Ga{\"{e}}l Varoquaux},
  title     = {{API} design for machine learning software: experiences from the scikit-learn
               project},
  booktitle = {ECML PKDD Workshop: Languages for Data Mining and Machine Learning},
  year      = {2013},
  pages = {108--122},

"""

"""
All the above Modules have their rights reserved.
@ author: Richard Lu
@ current version: 1.7
@ contact info: cchmantle@gmail.com
@ Initial Date: 2021/03/31

update log: 
@@ log 1: 1.0 --> 1.1
@ date: 2021/04/10
@ content: 1. now support random feature selection, use it in dataset.generator().
           2. AE double loss model has been added, please turn on the double_loss parameter in 
           TrainingProcess class when use.
           
@@ log 2: 1.1 --> 1.2
@ date: 2021/04/11
@ content: 1. add T-SNE support
           2. all the AE module has been added and tested.

@@ log 3: 1.2 --> 1.3
@ date: 2021/04/14
@ content: 1. Bayesian optimization for models added.

@@ log 4: 1.3 --> 1.4
@ date: 2021/04/17
@ content: 1. many bugs fixed, now we get best accuracy of the model and save the model.

@@ log 5: 1.4 --> 1.5
@ date: 2021/04/18
@ content: 1.many customized changes to new dataset, i.e.: Avila dataset.

@@ log 6: 1.5 --> 1.6
@ date: 2021/04/20
@ content: 1. add detailed plots for double loss AE

@@ log 7: 1.6 --> 1.7
@ date: 2021/04/21
@ content: 1. from now on, in training process, the Epoch 0 represent the initial model, and the real training 
            process starts from Epoch 2.
           2. many training related bug fixed.

"""


class AvilaDataset(object):
    """
    Costumed dataset for Avila.csv
    """

    def __init__(self, path='../Datasets/avila_clean.csv',
                 split=False, normalize=False):
        """
        :param path: full path for csv file.
        :param split: bool value, whether to split our dataset for training purpose.
        :param normalize: bool value, whether to normalize the generated dataset.
        """
        assert isinstance(path, str), "please enter a valid path"
        self.data_frame = pd.read_csv(path)
        self.tensor_check = False
        self.selection_check = None
        self.normalize = normalize
        self.g_called = 0
        self.pca = 'please generate data first and use PCA'
        self.singular = 'please generate data first and use SVD'
        if self.normalize:
            print("please make sure that there is no categorical feature value in the current data.")
        self.split = split
        if self.split:
            self.feature_train = 'Please generate data first'
            self.feature_test = 'Please generate data first'
            self.label_train = 'Please generate data first'
            self.label_test = 'Please generate data first'
            self.n_train = 'Please generate data first'
            self.n_test = 'Please generate data first'
        self.features = 'Please generate data first'
        self.labels = 'Please generate data first'
        self.n_input = 'please generate data first'
        self.n_output = 'please generate data first'

    def __len__(self):
        return len(self.data_frame)

    def generate(self, to_tensor=True, svd=None, pca_num=None, random_selection=None):
        features_frame = self.data_frame.drop(['Class'], axis=1)
        if random_selection:
            assert isinstance(random_selection, int) or random_selection > features_frame.shape[1], "please enter a " \
                                                                                                    "valid number for" \
                                                                                                    " selection"
            selected_feature_frame = features_frame.sample(frac=1, axis=1).iloc[:, :random_selection]
            self.selection_check = list(selected_feature_frame)
            self.features = selected_feature_frame.values
        else:
            self.features = features_frame.values
        self.labels = self.data_frame['Class'].values
        print(f"The {self.g_called + 1}th shuffled data has been generated.")

        # keep the label as a 1d array:
        self.labels = np.asarray(self.labels)

        if svd:
            assert pca_num is None, "please only use one method to transform data, for testing purpose," \
                                    "please call the corresponding function explicitly."
            if random_selection:
                assert svd <= random_selection, "please make sure your scd rank number is " \
                                                "smaller or equal to your random selected features"
            self.singular = self.SVD(svd)
            print(f"SVD transform finished, with {svd} rank kept")

        if pca_num:
            assert svd is None, "please only use one method to transform data, for testing purpose," \
                                "please call the corresponding function explicitly."
            if random_selection:
                assert pca_num <= random_selection, "please make sure your pca reduction number is " \
                                                    "smaller or equal to your random selected features"
            self.PCA_func(pca_num)
            print(f"PCA transform finished, reduced to {pca_num} dim now")

        if self.normalize:
            epsilon = 1e-8
            mean = np.mean(self.features, axis=0, keepdims=True).repeat(self.features.shape[0], axis=0)
            std = np.std(self.features, axis=0, keepdims=True).repeat(self.features.shape[0], axis=0)
            normalized_feature = (self.features - mean) / (std + epsilon)
            self.features = normalized_feature
            print("feature normalization complete.")

        if self.split:
            self.split_func()
            self.label_train = np.asarray(self.label_train)
            self.label_test = np.asarray(self.label_test)
            print("Data split finished.")

        self.n_input = 1 if self.features.ndim < 2 else self.features.shape[1]
        label_set = len(set(self.labels))
        self.n_output = label_set - 1 if label_set == 2 else label_set

        self.g_called += 1

        if to_tensor:
            self.tensor()

    def SVD(self, rank):
        assert not isinstance(self.features, str), "please generate data first"
        assert isinstance(rank, int), "please enter a valid integer rank number for SVD"
        """
        :param rank: how many ranks you want to keep
        :return: directly change the original feature matrix
        """
        u, s, vt = np.linalg.svd(self.features, full_matrices=False)
        s = np.diag(s)
        feature_approx = u[:, :rank] @ s[:rank, :rank] @ vt[:rank, :]
        self.features = feature_approx
        return s

    def eval_SVD(self):
        assert not isinstance(self.features, str), "please generate data first"
        assert not isinstance(self.singular, str), "please use SVD to transform the data first"
        s_array = np.diag(self.singular)
        s_array_norm = s_array / np.sum(s_array)
        order_list = np.arange(len(s_array)) + 1
        plt.subplot(1, 2, 1)
        plt.plot(order_list, s_array_norm)
        plt.title("Normalized Singular Values")
        plt.subplot(1, 2, 2)
        plt.plot(order_list, np.cumsum(s_array_norm))
        plt.title("Cumulated Singular Values")
        plt.show()

    def PCA_func(self, r_n):
        assert not isinstance(self.features, str), "please generate data first"
        assert isinstance(r_n, int), "please enter a valid integer reduction number for PCA"
        self.pca = PCA(r_n)
        self.features = self.pca.fit_transform(self.features)

    def eval_PCA(self):
        """
        it plots the percentage of variance explained by each principle component.
        """
        assert not isinstance(self.pca, str), "please use pca to generate data first"
        # variance and cumulative variance:
        plt.subplot(1, 2, 1)
        order_list = np.arange(len(self.pca.explained_variance_ratio_)) + 1
        plt.plot(order_list, self.pca.explained_variance_ratio_)
        plt.title("variance ")
        plt.subplot(1, 2, 2)
        cum_sum = np.cumsum(self.pca.explained_variance_ratio_)
        plt.plot(order_list, cum_sum)
        plt.title("cumulated variance ")
        plt.show()

    def split_func(self):
        assert not isinstance(self.features, str), "please generate data first"
        self.feature_train, self.feature_test, \
        self.label_train, self.label_test = train_test_split(self.features, self.labels,
                                                             test_size=0.4, shuffle=True)
        self.n_train = self.feature_train.shape[0]
        self.n_test = self.feature_test.shape[0]

    def AE_transform(self, epoch=500, h=8):
        """
        This is the function designed for AE training
        :return: it returns a model which is prepared for transfer learning.
        """
        assert self.g_called > 0, "please generate the data first"
        model_stage1 = AvilaAEModel(self.n_input, h)
        tr = TrainingProcess(model_stage1, self, ae_stage1=True)
        loss = nn.L1Loss()
        opt = t.optim.Adam(model_stage1.parameters(), lr=1e-2)
        tr.loss_opt(loss, opt)
        tr.training(epoch)
        tr.evaluation(loss_only=True, cm=False)
        return model_stage1

    def tensor(self):
        if self.g_called == 0:
            print("please generate data first")
            raise ValueError
        else:
            pass
        if self.tensor_check:
            if self.split:
                self.feature_train = self.feature_train.numpy()
                self.feature_test = self.feature_test.numpy()
                self.label_train = self.label_train.numpy()
                self.label_test = self.label_test.numpy()
            else:
                self.features = self.features.numpy()
                self.labels = self.labels.numpy()
            self.tensor_check = False
        else:
            if self.split:
                if self.feature_train.ndim < 2:
                    self.feature_train = self.feature_train.astype(np.float32).reshape((-1, 1))
                    self.feature_test = self.feature_test.astype(np.float32).reshape((-1, 1))
                else:
                    self.feature_train = self.feature_train.astype(np.float32)
                    self.feature_test = self.feature_test.astype(np.float32)
                self.label_train = self.label_train.astype(np.float32)
                self.label_test = self.label_test.astype(np.float32)
                self.feature_train = t.from_numpy(self.feature_train)
                self.feature_test = t.from_numpy(self.feature_test)
                self.label_train = t.from_numpy(self.label_train).type(t.LongTensor)
                self.label_test = t.from_numpy(self.label_test).type(t.LongTensor)
            else:
                self.labels = self.labels.astype(np.float32)
                if self.features.ndim < 2:
                    self.features = self.features.astype(np.float32).reshape((-1, 1))
                else:
                    self.features = self.features.astype(np.float32)
                self.features = t.from_numpy(self.features)
                self.labels = t.from_numpy(self.labels).type(t.LongTensor)
            self.tensor_check = True


class AvilaAEDoubleLoss(nn.Module):
    def __init__(self, n_input, h=9):
        super(AvilaAEDoubleLoss, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_input, 20),
            nn.ReLU(),
            nn.Linear(20, h),
            nn.ReLU(),
            nn.Linear(h, 20),
            nn.ReLU(),
            nn.Linear(20, n_input)
        )

        self.NN = AvilaModel(n_input)

    def forward(self, x):
        x = self.encoder(x)
        x = self.NN.forward(x)
        return x


class AvilaAESEDoubleLoss(nn.Module):
    def __init__(self, n_input, h=9):
        super(AvilaAESEDoubleLoss, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_input, 20),
            nn.ReLU(),
            nn.Linear(20, h),
            nn.ReLU(),
            nn.Linear(h, 20),
            nn.ReLU(),
            nn.Linear(20, n_input)
        )

        self.SENN = AvilaSEModel(n_input)

    def forward(self, x):
        x = self.encoder(x)
        x = self.SENN.forward(x)
        return x


class AvilaModel(nn.Module):
    """
    This is the baseline model for our task.
    """

    def __init__(self, n_input):
        """
        :param n_input: the # of input for the model, we now simply assume that the # of output is 1.(HR task)
        """
        super(AvilaModel, self).__init__()
        self.fc1 = nn.Linear(n_input, 20)
        self.fc2 = nn.Linear(20, 12)

    def forward(self, x):
        x = t.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AvilaAEModel(nn.Module):
    """
    This is the basic Auto-encoder model for our task.
    """

    def __init__(self, n_input, h=9):
        super(AvilaAEModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_input, 20),
            nn.ReLU(),
            nn.Linear(20, h)
        )

        self.decoder = nn.Sequential(
            nn.Linear(h, 20),
            nn.ReLU(),
            nn.Linear(20, n_input)
        )

    def forward(self, x):
        x = self.encoder(x)
        # remember to add an activation function here:
        x = t.relu(x)
        x = self.decoder(x)
        return x


class AEConverter(nn.Module):
    """
    This is the basic model for AE transfer learning.
    """

    def __init__(self, model: nn.Module, feature_extraction=False):
        super(AEConverter, self).__init__()

        self.encoder = list(model.children())[0]
        if feature_extraction:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.hidden = AvilaModel(self.encoder[-1].out_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.hidden.forward(x)
        return x


class AvilaSEModel(nn.Module):
    """
    This is the SENet model designed for HR dataset.
    """

    def __init__(self, n_input, h=20):
        """
        :param n_input: input feature dimension
        :param n: # of nodes for the first dense layer
        """
        super(AvilaSEModel, self).__init__()
        self.fc1 = nn.Linear(n_input, h)
        self.fc2 = nn.Linear(h, 12)
        self.SE = nn.Sequential(
            nn.Linear(n_input, 16),
            nn.ReLU(),
            nn.Linear(16, n_input),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.SE(x)
        x = t.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TrainingProcess(object):
    """
    This class is prepared for a standard NN training
    """

    def __init__(self, model, dataset: AvilaDataset, batch_size=256,
                 double_loss=False, weights=(6, 1), ae_stage1=False, ae_stage2=False):
        """
        :param model: define the training model object here.
        :param batch_size: define the training batch size for Mini-Batch SGD
        :param dataset: this parameter should be an object from the Dataset class containing a data generating function
        :param ae_stage1: this is a switch indicator for AE stage one, you won't need it to perform training since
                it's embedded in AE_transform() in dataset class
        :param ae_stage2: this is a switch indicator for AE stage two, you have to turn this on if you wish to
                train AE stage 2
        :param double_loss: this is a switch indicator for AE_double_loss class, you have to turn it on manually
                in order to train it
        """
        assert isinstance(batch_size, int), "please enter an integer number of" \
                                            "batch size"

        assert t.is_tensor(dataset.feature_train) and t.is_tensor(dataset.feature_test) and \
               t.is_tensor(dataset.label_train) and t.is_tensor(dataset.label_test), \
            "please make sure all the input data is in tensor form."

        assert isinstance(dataset.n_train, int) and isinstance(dataset.n_test, int), "please enter an " \
                                                                                     "integer training " \
                                                                                     "and " \
                                                                                     "testing number"
        assert dataset.feature_test.shape[0] >= batch_size, \
            "please make sure that the sample number in" \
            "both training and testing sets >= batch size"

        if double_loss:
            assert isinstance(model, AvilaAEDoubleLoss) or isinstance(model, AvilaAESEDoubleLoss), \
                "please build the model from AvilaAEDoubleLoss."
            assert isinstance(weights, tuple), "please enter a valid weights for AE double loss mode."
            self.dl_output = None
            self.hidden = None
            self.weights = weights
        self.dl = double_loss
        if ae_stage2:
            assert isinstance(model, AEConverter), "please build the model from AEConverter"
            self.hidden = None
        self.ae_stage2 = ae_stage2
        self.ae_stage1 = ae_stage1
        self.model = model
        self.general_hook_output = None
        self.epochs = "please finish the training first, i.e. call training()"
        self.batch_size = batch_size
        # variables for recording:
        self.best_model_idx = "please finish the training first, i.e. call training()"
        self.best_model = "please finish the training first, i.e. call training()"
        self.r_train_loss = "please finish the training first, i.e. call training()"
        self.r_test_loss = "please finish the training first, i.e. call training()"
        self.r_train_acc = "please finish the training first, i.e. call training()"
        self.r_test_acc = "please finish the training first, i.e. call training()"
        if double_loss:
            self.r_test_loss_recon = "please finish the training first, i.e. call training()"
            self.r_test_loss_classify = "please finish the training first, i.e. call training()"
            self.r_train_loss_recon = "please finish the training first, i.e. call training()"
            self.r_train_loss_classify = "please finish the training first, i.e. call training()"
        # data:
        self.dataset = dataset
        self.p_train = "please train the model first"
        self.p_test = "please train the model first"
        # loss function for Avila task is fixed as CrossEntropy
        # optimizer is fixed as Adam:
        self.criterion = nn.CrossEntropyLoss()
        self.opt = t.optim.Adam(self.model.parameters(), lr=1e-2)
        self.__one_step_switch = False
        self.train_complete = False
        self.n_batch_train = (dataset.n_train // self.batch_size) + 1
        self.n_batch_test = (dataset.n_test // self.batch_size) + 1
        print("data loading success.")

    def hook_double_loss_recon(self):
        assert self.dl, "please only use it in double_loss mode"

        def hook(module, inputs, outputs):
            self.dl_output = outputs

        return hook

    def hook_hidden(self):
        assert self.dl or self.ae_stage2 or self.ae_stage1, "please only use it in AE mode"

        def hook(module, inputs, outputs):
            self.hidden = outputs.detach().numpy()

        return hook

    def hook_general(self):

        def hook(module, inputs, outputs):
            self.general_hook_output = outputs.detach().numpy()

        return hook

    def clear_history(self):
        if not self.__one_step_switch:
            return "please do not use this function alone, call training()"
        self.opt.zero_grad()
        self.best_model = None
        self.best_model_idx = None
        self.r_train_loss = np.zeros(self.epochs)
        self.r_test_loss = np.zeros(self.epochs)
        self.r_train_acc = np.zeros(self.epochs)
        self.r_test_acc = np.zeros(self.epochs)
        if self.dl:
            self.r_test_loss_recon = np.zeros(self.epochs)
            self.r_test_loss_classify = np.zeros(self.epochs)
            self.r_train_loss_recon = np.zeros(self.epochs)
            self.r_train_loss_classify = np.zeros(self.epochs)

    def loss_opt(self, loss, opt: t.optim):
        """
        please change the optimizer and loss function here.
        """
        print("warning: you are changing the default loss function and optimizer!")
        self.criterion = loss
        self.opt = opt

    def double_loss(self, output_encoder, target_encoder,
                    output_final, target_final):
        """
        specifically designed for double loss ae model
        :param target_final: loss target for encoder layer
        :param target_encoder: loss target for the final classification task, i.e. batch labels
        :param output_final: final output of classification
        :param output_encoder: output from the encoder
        """
        weights = self.weights
        loss_func_1 = nn.L1Loss()
        loss_func_2 = nn.CrossEntropyLoss()
        loss_func_1 = loss_func_1(output_encoder, target_encoder)
        loss_func_2 = loss_func_2(output_final, target_final)

        loss_func = weights[0] * loss_func_1 + weights[1] * loss_func_2
        return loss_func, weights[0] * loss_func_1, weights[1] * loss_func_2

    def get_acc(self, inputs, targets):
        assert self.train_complete or self.__one_step_switch, "please first finish training"
        with t.no_grad():
            predicted = self.model(inputs)
            if self.ae_stage1:
                p_x = t.round(predicted)
            else:
                _, p_x = t.max(predicted, 1)
            p_x = p_x.numpy()
            targets = targets.numpy()
            acc = np.mean(targets == p_x)
        return acc

    def get_loss_dl(self, inputs, targets):
        assert self.train_complete or self.__one_step_switch, "please first finish training"
        assert self.dl, "please only use it in AE mode"
        with t.no_grad():
            predicted = self.model(inputs)
            output_encoder = self.dl_output
            _, loss_recon, loss_classify = self.double_loss(output_encoder, inputs, predicted, targets)
            loss_recon = loss_recon.item()
            loss_classify = loss_classify.item()
        return loss_recon, loss_classify

    def get_loss_acc(self, inputs, targets):
        assert self.train_complete or self.__one_step_switch, "please first finish training"
        with t.no_grad():
            predicted = self.model(inputs)
            if self.dl:
                output_encoder = self.dl_output
                loss, _, _ = self.double_loss(output_encoder, inputs, predicted, targets)

            elif self.ae_stage1:
                loss = self.criterion(predicted, inputs)

            else:
                loss = self.criterion(predicted, targets)
            loss_detach = loss.item()

            if self.ae_stage1:
                p_x = t.round(predicted)
            else:
                _, p_x = t.max(predicted, 1)
            p_x = p_x.numpy()
            targets = targets.numpy()
            acc = np.mean(targets == p_x)
        return loss_detach, acc

    def get_p_value_for_best(self, inputs):
        assert self.train_complete or self.__one_step_switch, "please first finish training"
        with t.no_grad():
            predicted = self.best_model(inputs)
            if self.ae_stage1:
                p_x = t.round(predicted)
            else:
                _, p_x = t.max(predicted, 1)
            p_x = p_x.numpy()
        return p_x

    def one_step(self, inputs, targets):
        if not self.__one_step_switch:
            return "please do not use one step function alone, call training()"
        # training:
        self.opt.zero_grad()
        outputs = self.model(inputs)
        if self.dl:
            output_encoder = self.dl_output
            loss, _, _ = self.double_loss(output_encoder, inputs, outputs, targets)
        elif self.ae_stage1:
            loss = self.criterion(outputs, inputs)
        else:
            loss = self.criterion(outputs, targets)
        loss.backward()
        self.opt.step()
        return loss

    def training(self, epochs=500, mode='SGD', record_only=False):
        """
        Here is where the main training takes place.
        :param record_only: don't print anything during the training
        :param epochs: define the total iteration number.
        :param mode: we have 2 modes here; GD: gradient descent; SGD: Mini-Batch stochastic gradient descent;
        :return: None
        """
        assert mode in ['GD', 'SGD'], "please enter a valid mode: 'GD' or 'SGD'."

        # define hooks:
        if self.dl:
            self.model.encoder.register_forward_hook(self.hook_double_loss_recon())

        if isinstance(self.model, AvilaSEModel):
            self.model.SE.register_forward_hook(self.hook_general())

        if isinstance(self.model, AvilaAESEDoubleLoss):
            self.model.SENN.SE.register_forward_hook(self.hook_general())

        self.__one_step_switch = True
        self.epochs = epochs

        if mode == 'GD':
            if not record_only:
                print("you're now in GD mode.")
            self.clear_history()
            for it in range(self.epochs):
                if it > 0:
                    # training:
                    self.model.train()
                    loss = self.one_step(self.dataset.feature_train, self.dataset.label_train)
                    self.r_train_loss[it] = loss.item()
                    self.r_train_acc[it] = self.get_acc(self.dataset.feature_train, self.dataset.label_train)
                    if self.dl:
                        loss_recon, loss_classify = self.get_loss_dl(self.dataset.feature_train,
                                                                     self.dataset.label_train)
                        self.r_train_loss_recon[it] = loss_recon
                        self.r_train_loss_classify[it] = loss_classify
                # testing:
                self.model.eval()
                loss_detach_test, acc_test = self.get_loss_acc(self.dataset.feature_test, self.dataset.label_test)
                self.r_test_loss[it] = loss_detach_test
                self.r_test_acc[it] = acc_test
                if self.dl:
                    loss_recon, loss_classify = self.get_loss_dl(self.dataset.feature_test, self.dataset.label_test)
                    self.r_test_loss_recon[it] = loss_recon
                    self.r_test_loss_classify[it] = loss_classify
                if it == np.argmax(self.r_test_acc):
                    self.best_model_idx = it
                    self.best_model = copy.deepcopy(self.model)
                    if isinstance(self.model, AvilaSEModel) or isinstance(self.model, AvilaAESEDoubleLoss):
                        setattr(self.best_model, "SE_weights", np.mean(self.general_hook_output, axis=0))
                    if self.dl or self.ae_stage2:
                        setattr(self.best_model, "hidden", self.hidden)
                    t.save(self.best_model.state_dict(), f"../Avila_project/Best_Models/"
                                                         f"best_model_of_{self.best_model.__class__.__name__}.pth")

                if (it + 1) % 50 == 0:
                    if not record_only:
                        print(f"The {it + 1}/{epochs} iteration of mode: {mode} is complete, "
                              f"with [training loss]: {self.r_train_loss[it]}, [training acc]: {self.r_train_acc[it]}, "
                              f"[testing loss]: {self.r_test_loss[it]}, [testing acc]: {self.r_test_acc[it]}")
            # get p value for the best model in GD:
            self.p_train = self.get_p_value_for_best(self.dataset.feature_train)
            self.p_test = self.get_p_value_for_best(self.dataset.feature_test)
            if not record_only:
                print("----------------------------------------------------")
                print(f"The training process of mode: {mode} is complete, "
                      f"with  [training loss]: {self.r_train_loss[self.best_model_idx]}, "
                      f" [training acc]: {self.r_train_acc[self.best_model_idx]}, \n"
                      f"Best [testing loss]: {self.r_test_loss[self.best_model_idx]},"
                      f" Best [testing acc]: {self.r_test_acc[self.best_model_idx]}")
        else:
            if not record_only:
                print("you're now in SGD mode, "
                      "please choose an appropriate Epoch number to ensure a better performance")
            self.clear_history()
            for it in range(self.epochs):
                if it > 0:
                    # training batch:
                    self.model.train()
                    train_loss = []
                    train_acc = []
                    if self.dl:
                        train_loss_recon = []
                        train_loss_classify = []
                    for i in range(self.n_batch_train):
                        x_train_batch, y_train_batch = (self.dataset.feature_train
                                                        [i * self.batch_size: (i + 1) * self.batch_size, :],
                                                        self.dataset.label_train
                                                        [i * self.batch_size: (i + 1) * self.batch_size])
                        loss = self.one_step(x_train_batch, y_train_batch)
                        train_loss.append(loss.item())
                        train_acc.append(self.get_acc(x_train_batch, y_train_batch))
                        if self.dl:
                            loss_recon, loss_classify = self.get_loss_dl(x_train_batch, y_train_batch)
                            train_loss_recon.append(loss_recon)
                            train_loss_classify.append(loss_classify)
                    self.r_train_loss[it] = np.mean(train_loss)
                    self.r_train_acc[it] = np.mean(train_acc)
                    if self.dl:
                        self.r_train_loss_recon[it] = np.mean(train_loss_recon)
                        self.r_train_loss_classify[it] = np.mean(train_loss_classify)
                    if not record_only:
                        if (it + 1) % 20 == 0:
                            print(f"The {it + 1}/{epochs} [training batch] of mode: {mode} is complete, "
                                  f"with [avg training loss]: {self.r_train_loss[it]}, "
                                  f"[avg training acc]: {self.r_train_acc[it]}")
                # testing batch:
                self.model.eval()
                test_loss = []
                test_acc = []
                if self.dl:
                    test_loss_recon = []
                    test_loss_classify = []
                for i in range(self.n_batch_test):
                    x_test_batch, y_test_batch = (self.dataset.feature_test
                                                  [i * self.batch_size: (i + 1) * self.batch_size, :],
                                                  self.dataset.label_test
                                                  [i * self.batch_size: (i + 1) * self.batch_size])
                    if self.dl:
                        loss_recon, loss_classify = self.get_loss_dl(x_test_batch, y_test_batch)
                        test_loss_recon.append(loss_recon)
                        test_loss_classify.append(loss_classify)
                    loss_detach_test, acc_test = self.get_loss_acc(x_test_batch, y_test_batch)
                    test_loss.append(loss_detach_test)
                    test_acc.append(acc_test)
                self.r_test_loss[it] = np.mean(test_loss)
                self.r_test_acc[it] = np.mean(test_acc)
                if self.dl:
                    self.r_test_loss_recon[it] = np.mean(test_loss_recon)
                    self.r_test_loss_classify[it] = np.mean(test_loss_classify)
                if it == np.argmax(self.r_test_acc):
                    self.best_model_idx = it
                    self.best_model = copy.deepcopy(self.model)
                    if isinstance(self.model, AvilaSEModel) or isinstance(self.model, AvilaAESEDoubleLoss):
                        setattr(self.best_model, "SE_weights", np.mean(self.general_hook_output, axis=0))
                    if self.dl or self.ae_stage2:
                        setattr(self.best_model, "hidden", self.hidden)
                    t.save(self.best_model.state_dict(), f"../Avila_project/Best_Models/"
                                                         f"best_model_of_{self.best_model.__class__.__name__}.pth")
                if not record_only:
                    if (it + 1) % 20 == 0:
                        print(f"The {it + 1}/{epochs} [testing batch] of mode: {mode} is complete, "
                              f"with [avg testing loss]: {self.r_test_loss[it]}, [avg testing acc]: {self.r_test_acc[it]}")

            self.p_train = self.get_p_value_for_best(self.dataset.feature_train)
            self.p_test = self.get_p_value_for_best(self.dataset.feature_test)
            if not record_only:
                print(f"The training process of mode: {mode} is complete, "
                      f"with [avg training loss]: {self.r_train_loss[self.best_model_idx]}, "
                      f"[avg training acc]: {self.r_train_acc[self.best_model_idx]}, \n"
                      f"Best [avg testing loss]: {self.r_test_loss[self.best_model_idx]}, "
                      f"Best [avg testing acc]: {self.r_test_acc[self.best_model_idx]}")

        self.__one_step_switch = False
        self.train_complete = True

    # plot confusion matrix:
    def plot_confusion_matrix(self, mode='test',
                              normalized=False,
                              title='Confusion Matrix',
                              colormap='GnBu'):
        assert self.train_complete, "please complete the training first"
        assert mode in ['train', 'test'], "please enter a valid mode, 'train' or 'test' "
        if mode == 'train':
            con_matrix = confusion_matrix(self.dataset.label_train.numpy(), self.p_train)
        else:
            con_matrix = confusion_matrix(self.dataset.label_test.numpy(), self.p_test)
        if normalized:
            matrix = con_matrix.astype('float') / np.expand_dims(con_matrix.sum(axis=1), -1)
            print('Confusion matrix has been normalized')
            print('-------------------------------------\n')
        else:
            print('Confusion matrix(Without normalized)')
            print('-------------------------------------\n')

        classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'W', 'X', 'Y']
        plt.imshow(con_matrix, interpolation='nearest', cmap=colormap)
        plt.title(title)
        plt.clim(0, 600)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes, rotation=45)
        # define the format of number in matrix:
        fmt = '.2f' if normalized else 'd'
        # create a threshold for each element in matrix:
        thresh = con_matrix.max() / 7
        for i, j in product(range(con_matrix.shape[0]), range(con_matrix.shape[1])):
            plt.text(j, i, format(con_matrix[i, j], fmt),
                     horizontalalignment='center',
                     color='white' if con_matrix[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.xlabel('predicted label')
        plt.ylabel('True label')

    def T_SNE_plot(self, n=2):
        assert self.train_complete and (self.dl or self.ae_stage2), "please complete the training first with AE mode"
        assert isinstance(n, int), "please enter a valid integer rank number for T_SNE"
        if self.dl:
            self.best_model.encoder[2].register_forward_hook(self.hook_hidden())
        elif self.ae_stage2 or self.ae_stage1:
            self.best_model.encoder[2].register_forward_hook(self.hook_hidden())
        else:
            self.best_model.fc1.register_forward_hook(self.hook_hidden())
        t_sne = TSNE(n_components=n, random_state=2)
        with t.no_grad():
            self.best_model(self.dataset.feature_test)
            hidden_data = self.hidden
        print("please wait, T-SNE transformation may take a while.")
        visualize_data = t_sne.fit_transform(hidden_data)
        plt.scatter(visualize_data[:, 0], visualize_data[:, 1],
                    c=self.dataset.label_test, alpha=0.5)
        plt.colorbar()
        plt.show()

    def evaluation(self, acc_only=False, loss_only=False, cm=True, record=False):
        assert self.train_complete, "please complete the training first"
        r_train_acc_from_1 = self.r_train_acc[1:]
        r_train_loss_from_1 = self.r_train_loss[1:]
        num_list_from_1 = list(np.arange(len(r_train_acc_from_1)) + 1)
        if acc_only:
            plt.plot(num_list_from_1, r_train_acc_from_1, label='training avg acc')
            plt.plot(self.r_test_acc, label='testing avg acc')
            plt.ylabel('accuracy')
            plt.title(f'train acc: {self.r_train_acc[self.best_model_idx]:.4f}, '
                      f'Best test acc: {self.r_test_acc[self.best_model_idx]:.4f}')
        elif loss_only:
            plt.plot(num_list_from_1, r_train_loss_from_1, label='training avg loss')
            plt.plot(self.r_test_loss, label='testing avg loss')
            plt.ylabel('losses')
            plt.title(f'train loss: {self.r_train_loss[self.best_model_idx]:.4f}, '
                      f'Best test loss: {self.r_test_loss[self.best_model_idx]:.4f}')
        else:
            plt.plot(num_list_from_1, r_train_acc_from_1, label='training avg acc')
            plt.plot(num_list_from_1, r_train_loss_from_1, label='training avg loss')
            plt.plot(self.r_test_acc, label='testing avg acc')
            plt.plot(self.r_test_loss, label='testing avg loss')
            plt.ylabel('accuracy and losses')
            plt.title(f'train acc: {self.r_train_acc[self.best_model_idx]:.4f}, '
                      f'Best test acc: {self.r_test_acc[self.best_model_idx]:.4f}')
        plt.xlabel('Iteration times')
        plt.legend()
        if not record:
            plt.show()

        print('------------------------------------\n')
        print("(same model for train and test, but it's the Best for test)")
        print(f'train acc: {self.r_train_acc[self.best_model_idx]:.4f}, '
              f'Best test acc: {self.r_test_acc[self.best_model_idx]:.4f}')
        if cm and (not record):
            self.plot_confusion_matrix(mode='test', title='Confusion Matrix for Test set')
            plt.show()
        if record:
            return format(self.r_test_acc[self.best_model_idx], '.4f')

    def get_SE_weight(self):
        assert (isinstance(self.model, AvilaSEModel) or isinstance(self.model, AvilaAESEDoubleLoss)) \
               and self.train_complete, "please complete the training first or " \
                                        "choose a valid model (SE model)"
        feature_array = np.asarray(list(self.dataset.data_frame))
        np.delete(feature_array, [-1])
        weights = self.best_model.SE_weights
        idx = np.argsort(-weights)
        sorted_weights = t.from_numpy(weights[idx]).type(t.float64)
        sorted_weights_pc = (softmax(sorted_weights, dim=0).numpy()) * 100
        sorted_feature = feature_array[idx]
        print(f"sorted feature in descending order: \n {sorted_feature}")
        print("--------------------------------------")
        print(f"with corresponding value in percentage:\n :{np.round(sorted_weights_pc, 4)}")
        return sorted_feature, np.round(sorted_weights_pc, 4)

    def plot_dl_loss(self, record=False):
        assert self.dl, "please only use it in AE mode"
        assert self.train_complete, "please complete the training first"
        r_train_loss_recon_from_1 = self.r_train_loss_recon[1:]
        r_train_loss_classify_from_1 = self.r_train_loss_classify[1:]
        num_list_from_1 = list(np.arange(len(r_train_loss_recon_from_1)) + 1)
        plt.plot(num_list_from_1, r_train_loss_recon_from_1, label="reconstruction avg train loss")
        plt.plot(num_list_from_1, r_train_loss_classify_from_1, label="classification avg train loss")
        plt.plot(self.r_test_loss_recon, label="reconstruction avg test loss")
        plt.plot(self.r_test_loss_classify, label="classification avg test loss")
        plt.legend()
        if not record:
            plt.show()
        print("For the Best model, we have:")
        print("------------------------------------------------")
        print(f"Weighted reconstruction avg test loss:{self.r_train_loss_recon[self.best_model_idx]:.4f} \n"
              f"Weighted classification avg test loss:{self.r_test_loss_classify[self.best_model_idx]:.4f}")
        print(f"True reconstruction avg test loss:"
              f"{(self.r_train_loss_recon[self.best_model_idx] / self.weights[0]):.4f} \n"
              f"True classification avg test loss:"
              f"{(self.r_test_loss_classify[self.best_model_idx] / self.weights[1]):.4f}")
        print("------------------------------------------------")
        return format(self.r_train_loss_recon[self.best_model_idx], '.4f'), \
               format((self.r_train_loss_recon[self.best_model_idx] / self.weights[0]), '.4f')


# Note: if you don't want to use Bayesian optimization,
# please comment the following class and all the ax modules in the beginning for speeding up the program
class AvilaBO(object):
    """
    Bayesian optimization for models in Avila project
    """

    def __init__(self, model_class, dataset: AvilaDataset, ae_double=False, epoch_in_train=100):
        assert issubclass(model_class, nn.Module) and isinstance(dataset, AvilaDataset), "please enter " \
                                                                                         "a valid model and dataset"
        self.model_class = model_class
        self.model = None
        self.model_after_trail = None
        self.dataset = dataset
        self.ae_double = ae_double
        self.best_para = None
        self.values = None
        self.experiment = None
        self.epoch = epoch_in_train

    def func_eval(self, para: Dict[str, float]):
        if not self.ae_double:
            self.model = self.model_class(self.dataset.n_input)
            tr = TrainingProcess(self.model, self.dataset)
            tr.opt = t.optim.Adam(self.model.parameters(),
                                  lr=para.get("lr", 1e-2)
                                  )
            tr.training(epochs=self.epoch, record_only=False)

        else:
            self.model = self.model_class(self.dataset.n_input, h=15)
            weight = para.get("weights", 1)
            tr = TrainingProcess(self.model, self.dataset,
                                 weights=(1, weight), double_loss=True)
            tr.opt = t.optim.Adam(self.model.parameters(),
                                  lr=para.get("lr", 1e-2)
                                  )
            tr.training(epochs=self.epoch, record_only=False)

        return tr.r_test_acc[-1]

    def opt_b(self, total_trails=20, continuous=False):
        if not self.ae_double:
            parameters = [
                {"name": "lr",
                 "type": "range",
                 "bounds": [1e-5, 1e-2],
                 "log_scale": True}
            ]
        else:
            parameters = [
                {"name": "lr",
                 "type": "range",
                 "bounds": [1e-5, 1e-2],
                 "log_scale": True},

                {"name": "weights",
                 "type": "range",
                 "bounds": [1, 100],
                 "value_type": "float"}
            ]

        if continuous:
            self.best_para, self.values, self.experiment, self.model = optimize(
                parameters=parameters,
                evaluation_function=self.func_eval,
                objective_name='accuracy',
                total_trials=total_trails
            )
        else:
            self.best_para, self.values, self.experiment, self.model_after_trail = optimize(
                parameters=parameters,
                evaluation_function=self.func_eval,
                objective_name='accuracy',
                total_trials=total_trails
            )

        print(f"the best acc is: {self.values[0]}, \n"
              f"achieved at: {self.best_para}")

        if self.ae_double:
            if continuous:
                render(plot_contour(model=self.model,
                                    param_x='lr', param_y='weights', metric_name='accuracy'))
            else:
                render(plot_contour(model=self.model_after_trail,
                                    param_x='lr', param_y='weights', metric_name='accuracy'))


if __name__ == '__main__':
    pass
