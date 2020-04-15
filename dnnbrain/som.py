#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter
from warnings import warn
import matplotlib.pyplot as plt


"""
###############################################################################
Goal: SOM connected from DCNN to simulate the VTC
###############################################################################
"""

class VTCSOM(object):
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

    def random_weights_init(self, data):
        """
        Initializes the weights of the SOM picking random samples from data.
        """
        it = np.nditer(self._activation_map, flags=['multi_index'])
        while not it.finished:
            rand_i = self._random_generator.randint(len(data))
            self._weights[it.multi_index] = data[rand_i]
            it.iternext()

    def pca_weights_init(self, data):
        """Initializes the weights to span the first two principal components.
        This initialization doesn't depend on random processes and
        makes the training process converge faster.
        It is strongly reccomended to normalize the data before initializing
        the weights and use the same normalization for the training data.
        """
        pc_length, pc = np.linalg.eig(np.cov(np.transpose(data)))
        pc_order = np.argsort(-pc_length)
        for i, c1 in enumerate(np.linspace(-1, 1, len(self._neigx))):
            for j, c2 in enumerate(np.linspace(-1, 1, len(self._neigy))):
                self._weights[i, j] = c1*pc[pc_order[0]] + c2*pc[pc_order[1]]
                
    def Normalize_X(self, data):
        """Normalize the data, where data is like (1000samples, 64features)"""
        N, D = data.shape    # N is number of sample
        for i in range(N):
            temp = np.sum(np.multiply(data[i], data[i]))
            data[i] /= np.sqrt(temp)
        return data
                
                
    #### Training ####            
    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*np.pi*sigma*sigma
        ax = np.exp(-np.power(self._neigx-c[0], 2)/d)
        ay = np.exp(-np.power(self._neigy-c[1], 2)/d)
        return np.outer(ax, ay)  # the external product gives a matrix

    def _mexican_hat(self, c, sigma):
        """Mexican hat centered in c."""
        xx, yy = np.meshgrid(self._neigx, self._neigy)
        p = np.power(xx-c[0], 2) + np.power(yy-c[1], 2)
        d = 2*np.pi*sigma*sigma
        return np.exp(-p/d)*(1-2/d*p)

    def _bubble(self, c, sigma):
        """Constant function centered in c with spread sigma.
        sigma should be an odd value.
        """
        ax = np.logical_and(self._neigx > c[0]-sigma,
                            self._neigx < c[0]+sigma)
        ay = np.logical_and(self._neigy > c[1]-sigma,
                            self._neigy < c[1]+sigma)
        return np.outer(ax, ay)*1.

    def _triangle(self, c, sigma):
        """Triangular function centered in c with spread sigma."""
        triangle_x = (-abs(c[0] - self._neigx)) + sigma
        triangle_y = (-abs(c[1] - self._neigy)) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        return np.outer(triangle_x, triangle_y)

    def train(self, data, num_iteration):
        """Trains the SOM.
        data : np.array or list
            Data matrix.
        num_iteration : int
            Maximum number of iterations (one iteration per sample).
        """
        data = self.Normalize_X(data)
        iterations = np.arange(num_iteration) % len(data)
        self._random_generator.shuffle(iterations)
        
        for t, iteration in enumerate(tqdm(iterations)):
            self.update(data[iteration], 
                        self.winner(data[iteration]), 
                        t, num_iteration)

    def winner(self, x):
        """Compute the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return np.unravel_index(self._activation_map.argmin(), self._activation_map.shape)
        #return np.unravel_index(self._activation_map.argmax(), self._activation_map.shape)
            
    def _activate(self, x):
        s = np.subtract(x, self._weights)  # x - w
        self._activation_map = np.linalg.norm(s, axis=-1)
        #a,b,c = self._weights.shape[0],self._weights.shape[1],self._weights.shape[2]
        #self._activation_map = np.dot(x, self._weights.reshape(a*b, c).T).reshape(a,b)

    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons.
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        eta = self.decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sigma = self.decay_function(self._sigma, t, max_iteration)
        #sigma = self._sigma
        # improves the performances
        g = self.neighborhood(win, sigma) * eta
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += np.einsum('ij, ijk->ijk', g, x-self._weights)
        
    def decay_function(self, scalar, t, max_iter):
        return scalar / (1+t/(max_iter/2))
        

    #### Prediction ####
    def activation_map(self, x):
        """Returns the activation map to x, where x is a sample."""
        self._activate(x)
        return self._activation_map
    
    def winner_position(self, x):
        """Returns the winners' position of x, where x is a sample."""
        return self.winner(x)
    
    def winner_times(self, data):
        """Returns a matrix where the element i,j is the number of times
           that the neuron i,j have been winner, where data is a array."""
        a = np.zeros((self._weights.shape[0], self._weights.shape[1]))
        data = self.Normalize_X(data)
        for x in data:
            a[self.winner(x)] += 1
        return a

    def winners_dict(self, data):
        """Returns a dictionary wm where wm[(i,j)] is a list
        with all the patterns that have been mapped in the position i,j."""
        winmap = defaultdict(list)
        data = self.Normalize_X(data)
        for x in data:
            winmap[self.winner(x)].append(x)
        return winmap
    
    def labels_map(self, data, labels):
        """Returns a dictionary wm where wm[(i,j)] is a dictionary
           that contains the number of samples from a given label
           that have been mapped in position i,j. 
           Such as (48, 46): Counter({0.6276549594387113: 1})
        data : np.array or list
            Data matrix.
        label : np.array or list
            Labels for each sample in data.
        """
        winmap = defaultdict(list)
        data = self.Normalize_X(data)
        for x, label in zip(data, labels):
            winmap[self.winner(x)].append(label)
        for position in winmap:
            winmap[position] = Counter(winmap[position])
        return winmap
    
    
    #### Visulization ####
    def U_matrix(self):
        def distance_map(self):
            """Returns the distance map of the weights.
            Each cell is the normalised sum of the distances between
            a neuron and its neighbours."""
            def fast_norm(x):
                return np.sqrt(np.dot(x, x.T))
            um = np.zeros((self._weights.shape[0], self._weights.shape[1]))
            it = np.nditer(um, flags=['multi_index'])
            while not it.finished:
                for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                    for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                        if (ii >= 0 and ii < self._weights.shape[0] and
                                jj >= 0 and jj < self._weights.shape[1]):
                            w_1 = self._weights[ii, jj, :]
                            w_2 = self._weights[it.multi_index]
                            um[it.multi_index] += fast_norm(w_1-w_2)
                it.iternext()
            um = um/um.max()
            return um
        heatmap = distance_map(self)  #生成U-Matrix
        plt.figure(figsize=(7, 7))
        plt.title('U-Matrix')
        plt.imshow(heatmap, cmap=plt.get_cmap('bone_r'))
        plt.colorbar()
        
    def Component_Plane(self, feature_index):
        """Component_Plane表示了map里每个位置的神经元对什么特征最敏感
           (或者理解为与该特征取值最匹配)"""
        plt.figure(figsize=(7, 7))
        plt.title('Component Plane: feature_index is %d' % feature_index)
        plt.imshow(self._weights[:,:,feature_index], cmap='coolwarm')
        plt.colorbar()
        plt.show()
    
    def distance_from_weights(self, data):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        data = self.Normalize_X(data)
        input_data = np.array(data)
        weights_flat = self._weights.reshape(-1, self._weights.shape[2])
        input_data_sq = np.power(input_data, 2).sum(axis=1, keepdims=True)
        weights_flat_sq = np.power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = np.dot(input_data, weights_flat.T)
        return np.sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)

