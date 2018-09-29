import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
import sklearn
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from helper_function_obtain_data import *
import sys
import seaborn as sns
from scipy.interpolate import CubicSpline
import csv

class Embed:
	"""
	Implements time-embedding of a time series X

	"""
	def __init__(self, X):
		"""
		Input: X - a one dimensional time series
		"""

		self.X = X

	def embed_vectors(self, tau, E):
		"""
		Constructs shadow manifold where X lies in from X using delayed coordinates
		based on Taken's Embedding Theorems, see 
		http://www.scholarpedia.org/article/Attractor_reconstruction
		INPUTS: tau -- scalar, time delay
				E -- scalar, embedding dimension
		OUTPUT: shadow_manifold -- matrix of size (length of X - tau*(E-1), E)
								the attractor that X lies in
		"""

		N_ts = self.X.shape[0] # length of time series
		shadow_manifold = np.zeros((N_ts - tau*(E-1), E))

		for i in range(shadow_manifold.shape[0]):
			shadow_manifold[i,:] = self.X[i:i+tau*(E-1)+1:tau]

		return shadow_manifold

class ccm:
	"""
	Constructs \hat{x_t}|M_y, cross-mapped estimate of x(t) from historic record of y,
	which is contained in the shadow manifold M_y constructed from y(t),
	and vice versa
	See the introduction of cross convergent mapping (ccm) at
	Detecting Causality in Complex Ecosystems by Sugihara et al
	"""

	def __init__(self, E, tau, X, Y):
		"""
		INPUTS: tau -- scalar, time delay
				E -- scalar, embedding dimension
				X -- vector, time series
				Y -- vector, time series
		"""

		self.E = E
		self.tau = tau
		self.X = X
		self.Y = Y


	def sampling_with_replacement(self, Mx, My, L):
		"""
		Computes convergence of the correlation between observed valus and cross-mapped
		estimates as a function of the library lenth L
		say we have an embedded shadow manifold M_x of size (n, E), then we need to compute
		the correlation averaged over n_samples where each sample is of size (L, E) 
		hence we need to randomly sample L indices in 1:n non-repetively

		Args:
		- Mx: shadow manifold built by x
		- My: shadow manifold built by y
		Note the rows of Mx and My should be same
		- L: length of library extracted from the shadow manifolds Mx and My
		"""
		
		n = Mx.shape[0] # 

		self.indicesX = random.sample(range(n), L)
		self.Mx = Mx[self.indicesX, :]
		self.My = My[self.indicesX, :]

	def sampling_without_replacement(self, Mx, My, L):
		"""
		computing convergence of the correlation between observed valus and cross-mapped
		estimates as a function of the library lenth L
		say we have an embedded shadow manifold M_x of size (n, E), then we need to compute
		the correlation averaged over n_samples where each sample is of size (L, E) 
		hence we need to randomly sample L indices in 1:n non-repetively

		Args:
		- Mx: shadow manifold built by x
		- My: shadow manifold built by y
		Note the rows of Mx and My should be same
		- L: length of library extracted from the shadow manifolds Mx and My
		"""
		
		n = Mx.shape[0] # 

		self.indicesX = np.random.choice(range(n), L)
		self.Mx = Mx[self.indicesX, :]
		self.My = My[self.indicesX, :]

	
	def analysis(self, L):
		"""
		
		"""
		


		#########################################################################
		#########        construct cross-mapped estimate of y from Mx
		# find E+1 nearest neights of x_t in Mx
		# number of nearest neighbors is set to be E+2, because the closest of course is itself
		# and will be thrown out
		num_nn = self.E + 2 

		nbrsX = NearestNeighbors(n_neighbors = num_nn, algorithm = 'auto').fit(self.Mx)
		distances, indices = nbrsX.kneighbors(self.Mx)

		# order y_t based on indices given by the nearest neighbor algorithm
		indY = np.array(self.indicesX)

		yh_t = np.zeros((L, self.E+1))
		for i in range(yh_t.shape[1]):
			yh_t[:,i] = self.Y[indY[indices[:,i+1]]+(self.E-1)*self.tau]

		# compute weights
		Wy = np.zeros((L, self.E+1))
		for i in range(Wy.shape[0]):
			Wy[i,:] = np.exp(-distances[i, 1:]/(distances[i,1]+0.0001))
			Wy[i,:] = Wy[i,:]/np.sum(Wy[i,:]) # normalize

		# compute the estimate of y from Mx
		y_Mx = np.sum(Wy*yh_t, axis = 1)

		#########################################################################
		#########        construct cross-mapped estimate of x from My
		# find E+1 nearest neights of y_t in My
		# number of nearest neighbors is set to be E+2, because the closest of course is itself
		# and will be thrown out
		num_nn = self.E + 2 

		nbrsY = NearestNeighbors(n_neighbors = num_nn, algorithm = 'auto').fit(self.My)
		distances, indices = nbrsY.kneighbors(self.My)

		# order x_t based on indices given by the nearest neighbor algorithm
		indX = np.array(self.indicesX)

		xh_t = np.zeros((L, self.E+1))
		for i in range(xh_t.shape[1]):
			xh_t[:,i] = self.X[indX[indices[:,i+1]]+(self.E-1)*self.tau]

		# compute weights
		Wx = np.zeros((L, self.E+1))
		for i in range(Wx.shape[0]):
			Wx[i,:] = np.exp(-distances[i, 1:]/(distances[i,1]+0.0001))
			Wx[i,:] = Wx[i,:]/np.sum(Wx[i,:]) # normalize

		# compute the estimate of y from Mx
		x_My = np.sum(Wx*xh_t, axis = 1)

		# order the original time series for comparison
		x_ordered = self.X[indX+(self.E-1)*self.tau]
		y_ordered = self.Y[indY+(self.E-1)*self.tau]

		return y_Mx, x_My, x_ordered, y_ordered

class simplex:
    """
    Prediction using simplex projection introduced in 
    Nonlinear forecasting as a way of distinguishing chaos from measurement
    error in time series by Sugihara & May
    This is used to find optimal E and tau
    """

    def __init__(self, E, X, Mx, tau):
        self.E = E # embedding dimension
        self.tau = tau # delay
        self.X = X # one dimensional time series data
        self.Mx = Mx # shadow manifold constructed 


    def one_forecasting(self, Tp, predicting_ind):

        """
        Tp step ahead forecasting for one row in Mx (predicting_ind picks the row index)
        Input: Tp: represent how many time steps we want to predict in the future
               predicting_ind: which row in Mx that we aim to predict its future values
        """

        # find E+1 nearest neights of x_t in Mx
        num_nn = self.E + 1
        N = self.Mx.shape[0]
        
        # Note that the reason using every row in Mx except the last one is because we can't predict using
        # the last row.
        nbrsX = NearestNeighbors(n_neighbors = num_nn+1, algorithm = 'auto').fit(self.Mx[:-1,:])
        distances, indices = nbrsX.kneighbors(self.Mx[:-1,:])

        # remove the first nn, because the closest one is always itself
        # indices will have the shape (Mx.shape[0], num_nn)
        # ith row in indices (shape (1, num_nn)) contains the row number in Mx of its n.n.
        indices = indices[:,1:]


        # for i in range(N):
        for i in [predicting_ind]:
            y = self.Mx[i,:] # targeted data point to be predicted
            # extract nearest neighbor for y
            nn_ind = indices[i,:] # the row number in Mx of y's n.n.
            nn_y = self.Mx[nn_ind, :] # y's n.n., nn_y has shape (num_nn, self.E)
            # compute weights
            dist_y = distances[i,1:] # distance of num_nn many n.n. from y
            d = dist_y[0]
            epsilon = 0.0001 # avoid dividing by 0 when computing Weights
            Weights = np.exp(-dist_y/(d+epsilon)) # of shape (1, num_nn)
            Weights = Weights/np.sum(Weights) # normalization

            # extract the Tp step ahead points in each nearest neighbors of y
            pred_nn = np.zeros((num_nn, Tp)) # store points at Tp step ahead in n.n. of y
            for j in range(num_nn):
                # compute the index of the ith nearest neighbor 
                ind = nn_ind[j]
                # compute the data index of the last entry in nn_y(ind, :), the ind^th n.n. of y
                last_ind = ind + (self.E-1)*self.tau 
                pred_nn[j,:] = self.X[last_ind+1:last_ind+Tp+1]

            # compute the weighted prediction
            y_pred = np.matmul(Weights, pred_nn)
            # extract the actual Tp step ahead points of y
            last_ind_y = i + (self.E-1)*self.tau 
            y_actual_ind = range(last_ind+1,last_ind+Tp+1)
            
        # combine output dictionary
        cache = {'prediction': y_pred, 'actual data indices': y_actual_ind, 'nearest neighbors': nn_y,
                'nearest neighbors indices': nn_ind}
        
        
    def forecasting(self, Tp):

        """
        Tp step ahead forecasting for one row in Mx (predicting_ind picks the row index)
        Input: Tp: represent how many time steps we want to predict in the future
        """

        # find E+1 nearest neights of x_t in Mx
        num_nn = self.E + 1
        N = self.Mx.shape[0]
        
        # Note that the reason using every row in Mx except the last one is because we can't predict using
        # the last row.
        nbrsX = NearestNeighbors(n_neighbors = num_nn+1, algorithm = 'auto').fit(self.Mx[:-1,:])
        distances, indices = nbrsX.kneighbors(self.Mx[:-1,:])

        # remove the first nn, because the closest one is always itself
        # indices will have the shape (Mx.shape[0], num_nn)
        # ith row in indices (shape (1, num_nn)) contains the row number in Mx of its n.n.
        indices = indices[:,1:]

        
        y_actual_list = []
        y_pred_list = []
        for i in range(1,N-1,1):
            y = self.Mx[i,:] # targeted data point to be predicted
            # extract nearest neighbor for y
            nn_ind = indices[i,:] # the row number in Mx of y's n.n.
            nn_y = self.Mx[nn_ind, :] # y's n.n., nn_y has shape (num_nn, self.E)
            # compute weights
            dist_y = distances[i,1:] # distance of num_nn many n.n. from y
            d = dist_y[0]
            epsilon = 0.0001 # avoid dividing by 0 when computing Weights
            Weights = np.exp(-dist_y/(d+epsilon)) # of shape (1, num_nn)
            Weights = Weights/np.sum(Weights) # normalization

            # extract the Tp step ahead points in each nearest neighbors of y
            pred_nn = np.zeros((num_nn, Tp)) # store points at Tp step ahead in n.n. of y
            for j in range(num_nn):
                # compute the index of the ith nearest neighbor 
                ind = nn_ind[j]
                # compute the data index of the last entry in nn_y(ind, :), the ind^th n.n. of y
                last_ind = ind + (self.E-1)*self.tau 
                pred_nn[j,:] = self.X[last_ind+1:last_ind+Tp+1]

            # compute the weighted prediction
            y_pred = np.matmul(Weights, pred_nn)
            y_pred_list.append(y_pred)
            # extract indices the actual Tp step ahead points of y in X
            last_ind_y = i + (self.E-1)*self.tau 
            y_actual_ind = range(last_ind+1,last_ind+Tp+1)
            y_actual = self.X[y_actual_ind]
            y_actual_list.append(y_actual)
            
        # combine output dictionary
        cache = {'prediction list': y_pred_list, 'actual data list': y_actual_list}

        return cache

# Tp steps ahead prediction for one row in Mx
def prediction_one_point(state, tau, predicting_ind, E, interpolate=False, plotting=False):
    """  
    Visualize how nearest neighbors look like using SIMPLEX projection method
    E: embedding dimension
    predicting_ind: which row in Mx that we aim to predict its future values
    """
    ## Obtain data
    disease = 'rubella'
    t, locations, tot_data, data_dic = preprocess_Mexico_disease_data(disease)
    x = data_dic[state] # causal variable, to be constructed
    start_year = 1986
    start_year_num = start_year - 1985 - 1
    start_month = 1
    start_time_ind = start_year_num*12 + start_month - 1
    end_year = 2008
    end_year_num = end_year - 1985 - 1
    end_month = 12
    end_time_ind = end_year_num*12 + end_month - 1
    t = t[start_time_ind:end_time_ind:1]
    X = x[start_time_ind:end_time_ind:1] # chop the series at month/year

    # interpolate data using cubic spline
    if interpolate:
        dt = 0.4 # finer time step size
        t_finer = np.arange(0, len(X), dt) # finer time array
        cs = CubicSpline(t, X)
        X = cs(t_finer)
        t = t_finer
    
    Tp = 1 # how many time steps to forecast in the future
    # generate embedding using the 1D time series
    eX = Embed(X) 
    Mx = eX.embed_vectors(tau,E) # shadow manifold of x
    simplex_pred = simplex(E, X, Mx, tau)
    cache = simplex_pred.one_forecasting(Tp, predicting_ind)
    
    y_pred = cache['prediction']
    y_actual_ind = cache['actual data indices']
    y_actual = X[y_actual_ind]
    t_pred = t[y_actual_ind] # time plotted with y_actual and y_pred
    
    y = Mx[predicting_ind,:] # targeted data to be predicted Tp ahead
    ty = t[predicting_ind:predicting_ind + (E-1)*tau + 1:tau] # time plotted with y
    
    nn_ind = cache['nearest neighbors indices']
    nn_y = Mx[nn_ind, :] # y's nearest neighbors
    t_nn = np.array([t[ind:ind+(E-1)*tau+1:tau] for ind in nn_ind])   
    
    # plotting  
    if plotting:
        plt.figure(0)
        plt.plot(t, X)
        plt.plot(ty, y, 'bo', markersize = 10)
        for i in range(nn_y.shape[0]):
            plt.plot(t_nn[i,:], nn_y[i, :], '*', markersize = 15)

        plt.figure(1)
        plt.plot(y, 'bo-', markersize = 10)
        plt.plot(np.transpose(nn_y), '*-')
        plt.plot(np.array(range(Tp))+len(y), y_pred, 'r*', markersize = 20)
        plt.plot(np.array(range(Tp))+len(y), y_actual, 'ro', markersize = 20)
    return y_actual, y_pred

def prediction(state, disease, tau, E, interpolate=False, plotting=False):
    """  
    E: embedding dimension
    """
    ## Obtain data
    t, locations, tot_data, data_dic = preprocess_Mexico_disease_data(disease)
    x = data_dic[state] # causal variable, to be constructed
#     x = tot_data
    start_year = 1986
    start_year_num = start_year - 1985 - 1
    start_month = 1
    start_time_ind = start_year_num*12 + start_month - 1
    end_year = 1996
    end_year_num = end_year - 1985 - 1
    end_month = 12
    end_time_ind = end_year_num*12 + end_month - 1
    t = t[start_time_ind:end_time_ind:1]
    X = x[start_time_ind:end_time_ind:1] # chop the series at month/year

    # interpolate data using cubic spline
    if interpolate:
        dt = 0.5 # finer time step size
        t_finer = np.arange(0, len(X), dt) # finer time array
        cs = CubicSpline(t, X)
        X = cs(t_finer)
        t = t_finer
    
    Tp = 1 # how many time steps to forecast in the future
    # generate embedding using the 1D time series
    eX = Embed(X) 
    Mx = eX.embed_vectors(tau,E) # shadow manifold of x
    
    simplex_pred = simplex(E, X, Mx, tau)
    cache = simplex_pred.forecasting(Tp)
    truth = cache['actual data list']
    prediction = cache['prediction list']
    
    corr_coeff, p_value = scipy.stats.pearsonr(truth, prediction)
    
    
    # plotting  
    if plotting:
        plt.figure(0)
        plt.plot(t, truth)
        plt.plot(t, prediction)
        
    return corr_coeff


