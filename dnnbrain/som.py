#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4/16/2020
BrainSOM mapping AI to HumanBrain 
@author: Zhangyiyuan
"""

import sys
from tqdm import tqdm
from time import time
from datetime import timedelta
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
    sys.stdout.write(progress)
    beginning = time()
    sys.stdout.write(progress)
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m-i+1) * (time() - beginning)) / (i+1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i+1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100*(i+1)/m)
        progress += ' - {time_left} left '.format(time_left=time_left)
        sys.stdout.write(progress)

def fast_norm(x):
    return np.sqrt(np.dot(x, x.T))

def asymptotic_decay(scalar, t, max_iter):
    return scalar / (1+t/(max_iter/2))

def none_decay(scalar, t, max_iter):
    return scalar




class VTCSOM(minisom.MiniSom):
    #### Initialization ####
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5,
                 sigma_decay_function=asymptotic_decay, lr_decay_function=asymptotic_decay,
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
        if sigma >= x or sigma >= y:
            warn('Warning: sigma is too high for the dimension of the map.')

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
        self._lr_decay_function = lr_decay_function
        self._sigma_decay_function = sigma_decay_function

        neig_functions = {'gaussian': self._gaussian,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle',
                                     'bubble'] and divmod(sigma, 1)[1] != 0:
            warn('sigma should be an integer when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]
                
    def Normalize_X(self, x):
        temp = np.sum(np.multiply(x, x))
        x /= np.sqrt(temp)
        return x
    
    
    #### Training ####      
    def Train(self, data, num_iteration, step_len, verbose):
        """Trains the SOM.
        data : np.array Data matrix (sample numbers, feature numbers).
        num_iteration : Maximum number of iterations.
        """            
        random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), num_iteration,
                                              verbose, random_generator)
        
        q_error = np.array([])
        t_error = np.array([])
        for t, iteration in enumerate(tqdm(iterations)):
            self.update(data[iteration], 
                        self.winner(data[iteration]), 
                        t, num_iteration) 
            if (t+1) % step_len == 0:
                q_error = np.append(q_error, self.quantization_error(data))
                t_error = np.append(t_error, self.topographic_error(data))
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
            print(' topographic error:', self.topographic_error(data)) 
        return q_error, t_error
    
    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        x = self.Normalize_X(x)
        s = np.subtract(x, self._weights)  # x - w
        self._activation_map = np.linalg.norm(s, axis=-1)

    def activate(self, x):
        """Returns the activation map to x."""
        self._activate(x)
        return self._activation_map
    
    def winner(self, x, k=0):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return np.unravel_index(self._activation_map.reshape(-1).argsort()[k],
                                self._activation_map.shape)
        
    def activation_response(self, data, k=0):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        self._check_input_len(data)
        a = np.zeros((self._weights.shape[0], self._weights.shape[1]))
        for x in data:
            a[self.winner(x, k)] += 1
        return a
    
    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons.
        Parameters
        ----------
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        eta = self._lr_decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._sigma_decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig)*eta
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += np.einsum('ij, ijk->ijk', g, x-self._weights)
    
    
    #### Visulization ####
    def U_matrix(self):
        heatmap = self.distance_map()
        plt.figure(figsize=(7, 7))
        plt.title('U-Matrix')
        plt.imshow(heatmap, cmap=plt.get_cmap('bone_r'))
        plt.colorbar()
        
    def Component_Plane(self, feature_index):
        """
        Component_Plane表示了map里每个位置的神经元对什么特征最敏感(或者理解为与该特征取值最匹配)
        """
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
    


if __name__ == '__main__':
    data = np.genfromtxt('/Users/mac/Desktop/TDCNN/minisom/examples/iris.csv', delimiter=',', usecols=(0, 1, 2, 3))
    # data normalization
    data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)

    # Initialization and training
    som = VTCSOM(7, 7, 4, sigma=3, learning_rate=0.5, 
                      sigma_decay_function=none_decay, lr_decay_function=asymptotic_decay,
                      neighborhood_function='gaussian')
    som.pca_weights_init(data)
    q_error, t_error = som.Train(data, 100, verbose=False)
    
    plt.figure()
    plt.plot(q_error)
    plt.ylabel('quantization error')
    plt.xlabel('iteration index')
    
    plt.figure()
    plt.plot(t_error)
    plt.ylabel('som.topographic error')
    plt.xlabel('iteration index')
    
    

