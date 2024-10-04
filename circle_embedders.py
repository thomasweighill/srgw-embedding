import numpy as np
from functools import partial
import matplotlib
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import sys, os
from scipy.spatial import distance_matrix
import networkx as nx
from itertools import product
from scipy.optimize import linear_sum_assignment as LSA
import random
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import tensorflow as tf
import tensorflow_probability as tfp
from ot.gromov import semirelaxed_gromov_wasserstein

sys.path.append('../CircularCoordinates/')
from circularcoordinates import CircCoordLn, weight_ft_0, weight_ft_with_degree_meta, weighted_circular_coordinate

def persistent_cohomology_coords(M):
    '''
    M is a distance matrix    
    
    returns a radius estimate, coordinates in [0,1] and the distortion 
    '''
    r_est = np.max(M) / np.pi
    y = weighted_circular_coordinate(M, distance_matrix=True)
    D = distortion(M, y, r_est)
    return r_est, y, [D]

def weighted_persistent_cohomology_coords(M):
    '''
    M is a distance matrix
    
    returns a radius estimate, coordinates in [0,1] and the distortion 
    '''
    r_est = np.max(M) / np.pi
    y = weighted_circular_coordinate(M, distance_matrix=True, weight_ft=weight_ft_0(5))
    D = distortion(M, y, r_est)
    return r_est, y, [D]

def circle_distances(coords, r):
    '''
    coords is a list of real numbers between 0 and 1
    r is a radius
    
    returns a distance matrix
    '''
    # compute geodesic distance matrix
    dec_pred = coords - np.floor(coords)
    inner_path = np.abs(np.subtract.outer(dec_pred, dec_pred))
    outer_path1 = np.add.outer(1-dec_pred, dec_pred)
    outer_path2 = np.add.outer(dec_pred, 1-dec_pred)
    distances = 2*np.pi*r*np.minimum.reduce([inner_path, outer_path1, outer_path2])
    return distances

def distortion(M, coords, r):
    '''
    M is a distance matrix
    coords is a list of real numbers between 0 and 1
    r is a radius
    
    returns the squared distortion
    '''
    M_pred = circle_distances(coords, r)
    return np.sum(np.square(M-M_pred))/len(M)**2
    
def distortion_tf(y_true, y_pred):
    '''
    compute the distortion for a particular embedding, used in optimization so tf functions are used
    
    y_true is the true NxN distance matrix
    y_pred is a 1x(N+1) vector whose first entry is the radius and later entries are circle coordinates
    '''
    p = y_pred[0,0]
    dec_pred = y_pred[:,1:] - tf.floor(y_pred[:,1:])
    
    inner_path = tf.abs(dec_pred - tf.transpose(dec_pred))
    outer_path1 = 1-dec_pred + tf.transpose(dec_pred)
    outer_path2 = tf.transpose(1-dec_pred) + dec_pred
    
    distances = 2*np.pi*p*tf.reduce_min([inner_path, outer_path1, outer_path2], axis=0)
    D = tf.square(distances-y_true)
    return tf.reduce_sum(D)/(y_pred.shape[1]-1)**2

def fit_to_circle(M, initial_x=None, initial_p=None, num_steps=2000, tol=1e-3, verbose=False, gamma=0.01, **kwargs):
    '''
    use gradient descent to find a locally optimal embedding into S^1
    
    M is a distance matrix
    initial_x is initial values for the circular coordinates
    initial_p is an initial radius value    
    
    returns a radius estimate, coordinates in [0,1] and distortion
    '''
    if initial_p is None:
        initial_p = np.max(M) / np.pi
    if initial_x is None:
        initial_x = list(np.random.random(size=(len(M))))
    initial = [initial_p]+list(initial_x)
    x = tf.Variable(np.array(initial).reshape(1,len(M)+1), shape=(1,len(M)+1))
        
    custom_loss = lambda: distortion_tf(M, x)
    opt = tf.keras.optimizers.Adam(learning_rate=gamma)
    losses = []
    for step in range(num_steps):
        losses.append(custom_loss())
        if verbose:
            print(losses[-1])
        opt.minimize(custom_loss, [x])
        if step > 1:
            if np.abs(losses[-1] - losses[-2]) < tol*losses[-2]:
                break
        if step == num_steps-1:
            print('WARNING: Did not converge')
            print('tolerance = ',  tol)
    p = x[0,0]
    xdec = x[:,1:] - tf.floor(x[:,1:])
    return p.numpy(), xdec.numpy()[0], losses

def fit_to_circle_ot_start(M, num_steps=2000, n_points=100, verbose=False, **kwargs):
    '''
    initialize gradient descent using a persistent cohomology hot-start
    
    M is a distance matrix  
    n_points is how many points to use on circle when coupling
    
    returns a radius estimate, coordinates in [0,1] and distortion
    '''
    r, y, losses = couple_to_circle(M, n_points=n_points)
    semiOT_coords = y
    if verbose:
        print('OT embedding completed.')
    return fit_to_circle(M, initial_x=semiOT_coords, num_steps=num_steps, verbose=verbose, **kwargs)

def plot_on_circle(coords, r=1, **kwargs):
    '''
    helpful function for plotting points on a circle based on coordinates
    '''
    put_on_circle = r*np.array([np.cos(2*np.pi*coords), np.sin(2*np.pi*coords)]).transpose()
    plt.gca().scatter(
        put_on_circle[:,0], put_on_circle[:,1], **kwargs
    )
    plt.gca().set_aspect(1)
    
def couple_to_circle(M, r=None, n_points=100, verbose=False):
    '''
    compute semi-relaxed GW matching to a discrete net on a circle
    
    M is a distance matrix
    r is a radius estimate (optional)
    
    returns a radius estimate, coordinates in [0,1] and distortion
    '''
    if r is None:
        r = np.max(M) / np.pi
    ts = np.linspace(0,1,n_points)+np.random.random(n_points)*1e-2 #used to be 1e-3
    D = circle_distances(ts, r) 
    h = np.ones(M.shape[0]) / M.shape[0]
    coupling, log = semirelaxed_gromov_wasserstein(M, D, h, symmetric=True,log=True,G0=None)
    argmonge = [
        np.argmax(coupling[i]) for i in range(coupling.shape[0])
    ]
    monge = [
        ts[a] for a in argmonge
    ]
    lowest_monge = np.argmin([
        np.max(coupling[i]) for i in range(coupling.shape[0])
    ])
    if verbose:
        top_two = np.argsort(-coupling[lowest_monge])[:2]
        print("Weakest coupling top two entries: ", top_two, coupling[lowest_monge][top_two])
        embedded_ds0 = np.array([D[top_two[0], argmonge[i]] for i in range(M.shape[0])])
        embedded_ds1 = np.array([D[top_two[1], argmonge[i]] for i in range(M.shape[0])])
        real_ds = M[lowest_monge]
        print("first choice: {:.3f}".format(np.linalg.norm(embedded_ds0-real_ds)))
        print("second choice: {:.3f}".format(np.linalg.norm(embedded_ds1-real_ds)))
    return r, monge, log['loss']

def couple_to_target(M, N, verbose=False, perturb=True, permstart=False, G0=None):
    '''
    compute semi-relaxed GW matching
    
    M is a distance matrix for the source
    N is a distance matrix for the target
    
    returns a list of indices assigning points in the source to points in the target
    '''
    h = np.ones(M.shape[0]) / M.shape[0]
    eps = np.min([
        N[i,j] for i in range(N.shape[0]) for j in range(N.shape[1]) if i!=j
    ])*1e-2
    if perturb:
        perturb = np.random.random(size=N.shape[0])*eps
        perturbedN = N + perturb + perturb.reshape(N.shape[0],1)
        np.fill_diagonal(perturbedN,0)
    else:
        perturbedN = N
    if permstart:
        initial_map = [
            np.random.choice(range(N.shape[0])) for i in range(M.shape[0])
        ]
        G0 = np.array(
            [[int(j==initial_map[i])/M.shape[0] for j in range(N.shape[0])] for i in range(M.shape[0])]
        )
    elif G0 is None:
        G0 = None
    if verbose:
        print(perturbedN)
    coupling, log = semirelaxed_gromov_wasserstein(
        M, perturbedN, h, symmetric=True,log=True,G0=G0, verbose=False, tol_abs=1e-14
    )
    if verbose:
        print(coupling)
        plt.plot(log['loss'][:50])
        plt.show()
        plt.plot(log['loss'])
        plt.show()
        print(log['loss'][-10:])
        plt.imshow(coupling)
        plt.show()
    argmonge = [
        np.argmax(coupling[i]) for i in range(coupling.shape[0])
    ]
    return argmonge

def fit_to_R2(M, initial_x=None, num_steps=2000, tol=1e-3, verbose=False, gamma=0.01, **kwargs):
    '''
    use gradient descent to find a locally optimal embedding into S^1
    
    M is a distance matrix
    initial_x is initial values for the circular coordinates
    
    returns coordinates and distortion
    '''
    if initial_x is None:
        initial_x = list(np.random.random(size=(len(M),2))*np.max(M))
    initial = list(initial_x)
    x = tf.Variable(np.array(initial).reshape(len(M),2), shape=(len(M),2))
    custom_loss = lambda: distortion_tf_R2(M, x)
    opt = tf.keras.optimizers.Adam(learning_rate=gamma)
    losses = []
    for step in range(num_steps):
        losses.append(custom_loss())
        if verbose:
            print(losses[-1])
        opt.minimize(custom_loss, [x])
        if step > 1:
            if np.abs(losses[-1] - losses[-2]) < tol*losses[-2]:
                break
        if step == num_steps-1:
            print('WARNING: Did not converge')
            print('tolerance = ',  tol)
    return x-tf.reduce_mean(x, axis=0), losses

def distortion_tf_R2(M, x):
    distances = tf.norm(
        x[:, None, :] - x[None, :, :],
        ord='euclidean',
        axis=-1
    )
    D = tf.square(distances-M)
    return tf.reduce_sum(D)/(M.shape[0])**2

def couple_to_R2(M, r=None, n_points=10, verbose=False):
    '''
    compute semi-relaxed GW matching to a bounded region of R^n
    
    M is a distance matrix
    r is a radius estimate (optional)
    
    returns coordinates
    '''
    if r is None:
        r = np.max(M)
    ts = np.linspace(-r,r,n_points)+np.random.random(n_points)*1e-2*r #used to be 1e-3
    us = np.linspace(-r,r,n_points)+np.random.random(n_points)*1e-2*r #used to be 1e-3
    points = np.array([
        [t,u] for t in ts for u in us
    ])
    D = distance_matrix(points, points) 
    h = np.ones(M.shape[0]) / M.shape[0]
    coupling, log = semirelaxed_gromov_wasserstein(M, D, h, symmetric=True,log=True,G0=None)
    argmonge = [
        np.argmax(coupling[i]) for i in range(coupling.shape[0])
    ]
    monge = np.array([
        points[a] for a in argmonge
    ])
    monge = monge - np.mean(monge, axis=0)
    return monge

def fit_to_R2_ot_start(M, num_steps=2000, n_points=10, verbose=False, **kwargs):
    '''
    initialize gradient descent using a persistent cohomology hot-start
    
    M is a distance matrix  
    n_points is how many points to use on each axis
    
    returns coordinates and distortion
    '''
    y = couple_to_R2(M, n_points=n_points)
    semiOT_coords = y + np.random.random(size=y.shape)*1e-2*np.max(np.abs(y))
    if verbose:
        print('OT embedding completed.')
    return fit_to_R2(M, initial_x=semiOT_coords, num_steps=num_steps, verbose=verbose, **kwargs)