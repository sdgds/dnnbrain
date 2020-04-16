#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4/16/2020
BrainSOM mapping AI to HumanBrain 
@author: Zhangyiyuan
"""

from tqdm import tqdm
from time import time
import numpy as np
from warnings import warn
import matplotlib.pyplot as plt
import minisom
import cv2



def _build_iteration_indexes(data_len, num_iterations,
                             verbose=False, random_generator=None):
    iterations = np.arange(num_iterations) % data_len
    if random_generator:
        random_generator.shuffle(iterations)
    if verbose:
        return _wrap_index__in_verbose(iterations)
    else:
        return iterations


def _wrap_index__in_verbose(iterations):
    m = len(iterations)
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    np.stdout.write(progress)
    beginning = time()
    np.stdout.write(progress)
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m-i+1) * (time() - beginning)) / (i+1)
        time_left = str(np.timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i+1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100*(i+1)/m)
        progress += ' - {time_left} left '.format(time_left=time_left)
        np.stdout.write(progress)


def fast_norm(x):
    return np.sqrt(np.dot(x, x.T))


def asymptotic_decay(scalar, t, max_iter):
    return scalar / (1+t/(max_iter/2))




class VTCSOM(minisom.MiniSom):
    #### Initialization ####
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5,
                 neighborhood_function='gaussian', random_seed=None):
        """
        x : int
            x dimension of the feature map.
        y : int
            y dimension of the feature map.
        input_len : int
            Number of the elements of the vectors in input.
        sigma : float
            Spread of the neighborhood function (sigma(t) = sigma / (1 + t/T) where T is num_iteration/2)
        learning_rate : 
            initial learning rate (learning_rate(t) = learning_rate / (1 + t/T)
        neighborhood_function : function, optional (default='gaussian')
            possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'
        random_seed : int
        """
        # Use seed to define the random generator
        self._random_generator = np.random.RandomState(random_seed)

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        
        # random initialization
        self._weights = self._random_generator.rand(x, y, input_len)*2-1
        self._weights /= np.linalg.norm(self._weights, axis=-1, keepdims=True)

        self._activation_map = np.zeros((x, y))
        self._neigx = np.arange(x)
        self._neigy = np.arange(y)  # used to evaluate the neighborhood function

        neig_functions = {'gaussian': self._gaussian,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle}
        if neighborhood_function in ['triangle','bubble'] and (divmod(sigma, 1)[1] != 0 or sigma < 1):
            warn('sigma should be an integer >=1 when triangle or bubble' +
                 'are used as neighborhood function')
        self.neighborhood = neig_functions[neighborhood_function]
                
    def Normalize_X(self, data):
        """Normalize the data, where data is like (1000samples, 64features)"""
        N, D = data.shape    # N is number of sample
        for i in range(N):
            temp = np.sum(np.multiply(data[i], data[i]))
            data[i] /= np.sqrt(temp)
        return data
    
    
    #### Training ####
    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        self._activation_map = (self._weights * x).sum(axis=2)

    def activate(self, x):
        """Returns the activation map to x."""
        self._activate(x)
        return self._activation_map
    
    def Train(self, data, num_iteration, initialization_way, verbose):
        """Trains the SOM.
        data : np.array or list
            Data matrix.
        num_iteration : int
            Maximum number of iterations (one iteration per sample).
        """
        if initialization_way == 'random':
            self.random_weights_init(data)
        if initialization_way == 'pca':
            self.pca_weights_init(data)
            
        data = self.Normalize_X(data)
        iterations = np.arange(num_iteration) % len(data)
        self._random_generator.shuffle(iterations)
        
        q_error = np.array([])
        t_error = np.array([])
        for t, iteration in enumerate(tqdm(iterations)):
            self.update(data[iteration], 
                        self.winner(data[iteration]), 
                        t, num_iteration) 
            q_error = np.append(q_error, self.quantization_error(data))
            t_error = np.append(q_error, self.topographic_error(data))
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
            print(' topographic error:', self.topographic_error(data)) 
            
        return q_error, t_error
    
    
    #### Visulization ####
    def U_matrix(self):
        heatmap = self.distance_map()
        plt.figure(figsize=(7, 7))
        plt.title('U-Matrix')
        plt.imshow(heatmap, cmap=plt.get_cmap('bone_r'))
        plt.colorbar()
        
    def Component_Plane(self, feature_index):
        plt.figure(figsize=(7, 7))
        plt.title('Component Plane: feature_index is %d' % feature_index)
        plt.imshow(self._weights[:,:,feature_index], cmap='coolwarm')
        plt.colorbar()
        plt.show()
        
    def Winners_map(self, data, blur=None):
        if blur == None:
            plt.figure()
            plt.imshow(self.activation_response(data))
            plt.colorbar()
        if blur == 'GB':
            img = self.activation_response(data)
            plt.figure()
            plt.imshow(cv2.GaussianBlur(img,(5,5),0))
            plt.colorbar()
    


                
