import numpy as np
import circle_embedders
import matplotlib.pyplot as plt
import mnist_loader
from scipy.spatial import distance_matrix
from PIL import Image 
from tqdm import tqdm
import pickle
import os
import time

def distortion(Mpred, Mtruth):
    return 1/2*(1/Mpred.shape[0])*np.sqrt(np.sum(np.square(Mpred-Mtruth)))

def rotate(image, deg):
    im = Image.fromarray((256*image.reshape(28,28)).astype(np.uint8))
    rotim = im.rotate(deg)
    return np.array(rotim).flatten()/256

train, test, valid = mnist_loader.load_data()

#adjust these lists
R2_methods = ['TSNE', 'PCA', 'MDS', 'SRGW_R2'] #full list is ['TSNE', 'PCA', 'MDS', 'SRGW_R2']
circle_methods = ['GD', 'SRGW', 'PCOH'] #full list is ['GD', 'SRGW', 'PCOH']

for digit in range(0,10):
    np.random.seed(2024)
    print(digit, '\n------ \n')
    train_images = np.vstack([train[0],test[0],valid[0]])
    train_labels = np.hstack([train[1],test[1],valid[1]])
    images = [image for i, image in enumerate(train_images) if train_labels[i]==digit]
    newimages = []
    newimages_labels = []
    for i, image in enumerate(images):
        j = np.random.randint(0,360)
        newimages_labels.append(0)
        newimages.append(
            rotate(image, 0)
        ) 
    if os.path.exists('MNIST_figures_unrotated/M{}.p'.format(digit)):
        #save time by loading the distance matrix if it exists
        M = pickle.load(open('MNIST_figures_unrotated/M{}.p'.format(digit), 'rb'))
    else:
        print('Computing M...')
        M = distance_matrix(newimages, newimages)
        pickle.dump(M, open('MNIST_figures_unrotated/M{}.p'.format(digit), 'wb'))
    X = {}
    y = {}
    r = {}
    times = {}
    
    for m in R2_methods+circle_methods:
        if m == 'PCA':
            from sklearn.decomposition import PCA
            np.random.seed(2024)
            print('PCA', end ='|')
            start_time = time.time()
            X['PCA'] = PCA(n_components=2).fit_transform(newimages)
            times['PCA'] = time.time() - start_time
        if m == 'TSNE':
            from sklearn.manifold import TSNE
            np.random.seed(2024)
            print('TSNE', end ='|')
            start_time = time.time()
            X['TSNE'] = TSNE(n_components=2, random_state=2024).fit_transform(np.array(newimages))
            times['TSNE'] = time.time() - start_time
        if m == 'Isomap':
            from sklearn.manifold import Isomap
            np.random.seed(2024)
            print('Isomap', end ='|')
            start_time = time.time()
            X['Isomap'] = Isomap(n_components=2).fit_transform(np.array(newimages))
            times['Isomap'] = time.time() - start_time
        if m == 'SRGW_R2':
            np.random.seed(2024)
            print('SRGW_R2', end ='|')
            start_time = time.time()
            X['SRGW_R2'], losses = circle_embedders.fit_to_R2_ot_start(M, gamma=0.1, n_points=20, verbose=True, tol=1e-4)
            times['SRGW_2'] = time.time() - start_time  
        if m == 'PCOH':
            print('PCOH', end ='|')
            np.random.seed(2024)
            start_time = time.time()
            r['PCOH'], y['PCOH'], losses = circle_embedders.weighted_persistent_cohomology_coords(M)
            X['PCOH'] = np.array(
                [[np.cos(2*np.pi*t), np.sin(2*np.pi*t)] for t in y['PCOH']]
            )
            times['PCOH'] = time.time() - start_time   
        if m == 'SRGW':
            print('SRGW', end ='|')
            np.random.seed(2024)
            start_time = time.time()
            r['SRGW'], y['SRGW'], losses = circle_embedders.fit_to_circle_ot_start(M, tol=1e-4)
            X['SRGW'] = np.array(
                [[np.cos(2*np.pi*t), np.sin(2*np.pi*t)] for t in y['SRGW']]
            )
            times['SRGW'] = time.time() - start_time    
        if m == 'GD':
            print('GD', end ='|')
            np.random.seed(2024)
            start_time = time.time()
            r['GD'], y['GD'], losses = circle_embedders.fit_to_circle(M, verbose=True, gamma=0.01, tol=1e-4)
            X['GD'] = np.array(
                [[np.cos(2*np.pi*t), np.sin(2*np.pi*t)] for t in y['GD']]
            )
            times['GD'] = time.time() - start_time  
            #run multiple trials if necessary
            # ds = [
            #     distortion(circle_embedders.circle_distances(y[m], r[m]), M)
            # ]
            # for trial in range(9):
            #     print('GD (trial {})'.format(trial))
            #     rr, yy, these_losses = circle_embedders.fit_to_circle(M)
            #     ds.append(
            #         distortion(circle_embedders.circle_distances(yy, rr), M)
            #     )
        if m == 'MDS':
            from sklearn.manifold import MDS
            print('MDS')
            np.random.seed(2024)
            start_time = time.time()
            X['MDS'] = MDS(n_components=2, dissimilarity='precomputed', eps=1e-4).fit_transform(M)
            times['MDS'] = time.time() - start_time

    #plot
    for m in R2_methods:
        plt.subplots(figsize=(5,5))
        plt.scatter(
            X[m][:,0],
            X[m][:,1],
            c='tab:blue', s=20, marker='x'
        )
        plt.gca().set_aspect(1)
        plt.savefig('MNIST_figures_unrotated/MNIST{}_{}.png'.format(digit, m), bbox_inches='tight', dpi=150)
        plt.close()
        
    for m in circle_methods:
        plt.subplots(figsize=(5,5))
        plt.scatter(
            X[m][:,0],
            X[m][:,1],
            c='tab:blue', s=20, marker='x'
        )
        plt.gca().set_aspect(1)
        plt.savefig('MNIST_figures_unrotated/MNIST{}_{}.png'.format(digit, m), bbox_inches='tight', dpi=150)
        plt.close()
    
    #distortions
    print('\n Distortion:')
    for m in R2_methods:
        print(m, end=' ')
        print('{:.3f}'.format(distortion(distance_matrix(X[m], X[m]), M)), end=' | ')

    for m in circle_methods:
        print(m, end=' ')
        print('{:.3f}'.format(
            distortion(circle_embedders.circle_distances(y[m], r[m]), M)
        ), end=' | ')
    print()
    #print('GD distortions over {} trials:'.format(len(ds)), sorted(ds)) 


    print('\n Times:')
    print(times)


