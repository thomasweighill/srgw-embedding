import random
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import sys, os
import networkx as nx
from itertools import product
import random
import tensorflow as tf
import tensorflow_probability as tfp
from ot.gromov import semirelaxed_gromov_wasserstein
import geopy
from geopy import distance
from math import radians, cos, sin, asin, sqrt

np.random.seed(2024)


pi_on_180 = 0.017453292519943295
pi = np.pi

def earth_distance(coords_1, coords_2):
    '''
    Distance as measure on the Earth using WGS-84
    '''
    return distance.geodesic(coords_1, coords_2).km

def sphere_distances_tf(coords, r=6371):
    '''
    Distance on a sphere
    '''
    rads = pi_on_180*coords
    lat1 = rads[:,1] - pi*tf.math.floor((rads[:,1]+pi/2)/pi)                     
    lng1 = rads[:,0] - 2*pi*tf.math.floor((rads[:,0]+pi)/(2*pi))         

    lat2 = rads[:,1] - pi*tf.math.floor((rads[:,1]+pi/2)/pi)                       
    lng2 = rads[:,0] - 2*pi*tf.math.floor((rads[:,0]+pi)/(2*pi))       

    diff_lat = lat1[:,None] - lat2
    diff_lng = lng1[:,None] - lng2
    d = tf.math.sin(diff_lat/2)**2 + tf.math.cos(lat1[:,None])*tf.math.cos(lat2)*tf.math.sin(diff_lng/2)**2
    return tf.cast(2 * 6371 * tf.math.asin(tf.math.sqrt(tf.nn.relu(d))), 'float32')

def couple_to_sphere(M, r=6371, n_points=10, verbose=False):
    '''
    Compute semi-relaxed GW matching to a discrete net on a sphere
    '''
    ts = tf.cast(np.array([
        [x,y] for x in np.linspace(-170,170,n_points) 
        for y in np.linspace(-80,80,n_points,endpoint=False)
    ]), dtype='float32')
    ts = ts + np.random.random(ts.shape)*1e-2
    D = sphere_distances_tf(ts, r).numpy() #these are distances
    print('Finding optimal coupling...')
    coupling, log = semirelaxed_gromov_wasserstein(
        M, D, np.ones(M.shape[0]) / M.shape[0], 
        symmetric=True,log=True,G0=None
    )
    argmonge = [
        np.argmax(coupling[i]) for i in range(coupling.shape[0])
    ]
    monge = np.array([
        ts.numpy()[a] for a in argmonge
    ])
    return r, monge, log['loss']


def sphere_distortion_tf(y_true, y_pred, r=6371):
    N = y_pred.shape[0]
    D_pred = sphere_distances_tf(y_pred)
    D = (D_pred-y_true)**2
    #print(np.min([D_pred[i,j] for i in range(D.shape[0]) for j in range(D.shape[1]) if i != j ]))
    return tf.reduce_sum(D)/(N**2)


def fit_to_sphere(
        M, initial_x=None, r=6371, 
        num_steps=2000, tol=1e-3, verbose=False, gamma=0.1, **kwargs
    ):
    '''
    Use gradient descent to find a locally optimal embedding into sphere
    '''
    N = len(M)
    if initial_x is None:
        initial_x = np.random.random(size=(len(M), 2))
        initial_x = initial_x*np.array([360,180]) - np.array([180,90])
    initial = list(initial_x)
    x = tf.Variable(np.array(initial).reshape(N,2), shape=(N,2), dtype='float32')  
    M_tf = tf.convert_to_tensor(M, dtype='float32')
    custom_loss = lambda: sphere_distortion_tf(M_tf, x)
    print(M_tf.dtype)
    opt = tf.keras.optimizers.Adam(learning_rate=gamma)
    losses = []
    coord_list = [] #
    for step in tqdm(range(num_steps)):  
        losses.append(custom_loss())
        if verbose:
            print(losses[-1])
        opt.minimize(custom_loss, [x])
        if step > 10:
            if np.abs(losses[-1] - losses[-2]) < tol*losses[-2]:
                break
        if step == num_steps-1:
            print('WARNING: Did not converge')
            print('tolerance = ',  tol)
    return x, losses

def fit_to_sphere_ot_start(M, num_steps=10000, n_points=10, **kwargs):
    '''
    Use SRGW+GD to embed to a sphere

    '''
    r, y, losses = couple_to_sphere(M, n_points=n_points)
    semiOT_coords = y + np.random.random(size=y.shape)
    return fit_to_sphere(M, initial_x=semiOT_coords, num_steps=num_steps, **kwargs)


def distortion(Mpred, Mtruth):
    return 1/2*(1/Mpred.shape[0])*np.sqrt(np.sum(np.square(Mpred-Mtruth)))


'''
Load the city data
'''
cities = pd.read_csv('simplemaps_worldcities_basicv1.77/worldcities.csv')
cities = cities.sort_values(by='population', ascending=False)
big_cities = cities[:20]
coords = np.array(
    [big_cities['lng'],
    big_cities['lat']]
).T
M = np.array(
[
    [
        earth_distance([x[1],x[0]],[y[1],y[0]]) for y in coords
    ] for x in coords
])

'''
Embed using t-SNE and MDS
'''
X = {}

from sklearn.manifold import TSNE
np.random.seed(2024)
X['TSNE'] = TSNE(n_components=3, metric='precomputed', perplexity=5, init='random').fit_transform(M)

from sklearn.manifold import MDS
np.random.seed(2024)
X['MDS'] = MDS(n_components=3, dissimilarity='precomputed').fit_transform(M)

print('TSNE', end=' ')
print('{:.3f}'.format(distortion(distance_matrix(X['TSNE'], X['TSNE']), M)))
print('MDS', end=' ')
print('{:.3f}'.format(distortion(distance_matrix(X['MDS'], X['MDS']), M)))

'''
Embed using GD
'''
np.random.seed(2024)
y, losses = fit_to_sphere(M)
print('GD', end=' ')
print(np.sqrt(losses[-1])/2)
# do 10 trials
# ds = [np.sqrt(losses[-1])/2]
# for trial in range(9):
#     y, losses = fit_to_sphere(M)
#     ds.append(np.sqrt(losses[-1])/2)
# print("GD distortions over 10 trials:", sorted(ds))


'''
Embed using SRGW
'''
np.random.seed(2024)
y, losses = fit_to_sphere_ot_start(M)
print('SRGW', end=' ')
print(np.sqrt(losses[-1])/2)


'''
Make the plots on the spheres. This requires some hard-coded hacks to make labels look good
'''
shift = {x:0 for x in big_cities['city']}
shift['Guangzhou'] = -2000
shift['Beijing'] = -1500
ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d', computed_zorder=False)
r = 6731*(1+0.01)


#rotations to apply for easy visualization -- first image
alpha = (-10)*pi_on_180
beta = (170)*pi_on_180
gamma = (-15)*pi_on_180

T = np.array([
    [np.cos(alpha), 0, np.sin(alpha) ],
    [0,1,0],
    [-np.sin(alpha), 0, np.cos(alpha) ]
])
R = np.array([
    [1,0,0 ],
    [0,np.cos(gamma),-np.sin(gamma)],
    [0,np.sin(gamma), np.cos(gamma) ]
])
S = np.array([
    [np.cos(beta), -np.sin(beta),0],
    [np.sin(beta), np.cos(beta),0],
    [0,0,1]
])

u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]

xxgrid = r*np.cos(u) * np.sin(v)
yygrid = r*np.sin(u) * np.sin(v)
zzgrid = r*np.cos(v)

new = R@T@S@np.array([xxgrid.flatten(), yygrid.flatten(), zzgrid.flatten()])
xxgrid, yygrid, zzgrid = new[0].reshape(30,20),  new[1].reshape(30,20), new[2].reshape(30,20)

ax.plot_surface(xxgrid, yygrid, zzgrid, cmap=plt.cm.gray, vmin=-12000)

aa = (y[:,0])*pi_on_180
bb = (y[:,1])*pi_on_180+np.pi/2
xx = r*np.sin(bb)*np.cos(aa)
yy = r*np.sin(bb)*np.sin(aa)
zz = r*np.cos(bb)

new = R@T@S@np.array([xx.flatten(), yy.flatten(), zz.flatten()])
xx, yy, zz = new[0],  new[1], new[2]

visible = [
    (xx[i]-yy[i] > 0)  for i in range(len(xx))
]
   
ax.scatter(
    [x for i, x in enumerate(xx) if visible[i]], 
    [y for i, y in enumerate(yy) if visible[i]], 
    [z for i, z in enumerate(zz) if visible[i]], 
    c='red', s=100
)


for i, (a,b) in enumerate(y):
    label = big_cities['city'][i]
    if visible[i]:
        ax.text(xx[i]+shift[label], yy[i]+shift[label], zz[i], label)

ax.set_aspect('equal')
    
ax.set_xlim(-r*1.1,r*1.1)
ax.set_ylim(-r*1.1,r*1.1)
ax.set_zlim(-r*1.1,r*1.1)


plt.savefig('sphere_on_sphere1.png', dpi=150, bbox_inches='tight')


#rotations to apply for easy visualization -- second image


ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d', computed_zorder=False)

r = 6731

alpha = (-10)*pi_on_180
beta = (260)*pi_on_180
gamma = (-15)*pi_on_180

T = np.array([
    [np.cos(alpha), 0, np.sin(alpha) ],
    [0,1,0],
    [-np.sin(alpha), 0, np.cos(alpha) ]
])
R = np.array([
    [1,0,0 ],
    [0,np.cos(gamma),-np.sin(gamma)],
    [0,np.sin(gamma), np.cos(gamma) ]
])
S = np.array([
    [np.cos(beta), -np.sin(beta),0],
    [np.sin(beta), np.cos(beta),0],
    [0,0,1]
])


u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]

xxgrid = r*np.cos(u) * np.sin(v)
yygrid = r*np.sin(u) * np.sin(v)
zzgrid = r*np.cos(v)

new = R@T@S@np.array([xxgrid.flatten(), yygrid.flatten(), zzgrid.flatten()])
xxgrid, yygrid, zzgrid = new[0].reshape(30,20),  new[1].reshape(30,20), new[2].reshape(30,20)

ax.plot_surface(xxgrid, yygrid, zzgrid, cmap=plt.cm.gray, vmin=-12000)

r = r*(1+0.01)

aa = (y[:,0])*pi_on_180
bb = (y[:,1])*pi_on_180+np.pi/2
xx = r*np.sin(bb)*np.cos(aa)
yy = r*np.sin(bb)*np.sin(aa)
zz = r*np.cos(bb)

new = R@T@S@np.array([xx.flatten(), yy.flatten(), zz.flatten()])
xx, yy, zz = new[0],  new[1], new[2]

visible = [
    (xx[i]-yy[i] > 0)  for i in range(len(xx))
]
   
ax.scatter(
    [x for i, x in enumerate(xx) if visible[i]], 
    [y for i, y in enumerate(yy) if visible[i]], 
    [z for i, z in enumerate(zz) if visible[i]], 
    c='red', s=100
)


for i, (a,b) in enumerate(y):
    label = big_cities['city'][i]
    if visible[i]:
        ax.text(xx[i], yy[i], zz[i], label)

ax.set_aspect('equal')
    
ax.set_xlim(-r*1.1,r*1.1)
ax.set_ylim(-r*1.1,r*1.1)
ax.set_zlim(-r*1.1,r*1.1)


plt.savefig('sphere_on_sphere2.png', dpi=150, bbox_inches='tight')
