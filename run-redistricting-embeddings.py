from gerrychain import Graph, Election, updaters, Partition, constraints, MarkovChain
from gerrychain.updaters import cut_edges
from gerrychain.random import random
from gerrychain.proposals import recom
from gerrychain.tree import recursive_tree_part
from gerrychain.accept import always_accept
from gerrychain import random
import numpy as np
import maup
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import pickle
import sys, os, datetime
from datetime import datetime
import networkx as nx
from itertools import product
from scipy.optimize import linear_sum_assignment as LSA
import random
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from circle_embedders import fit_to_circle_ot_start
np.random.seed(2022)
random.seed(2022)


level = 'blockgroups' #options are: blockgroups, blocks (only if you have the data available)
pop_col = 'TOTPOP'
tag = 'vanilla'


def pairwise_hamming(partitions):
    graph = partitions[0].graph
    A = np.array([
        [
            p.assignment[i] for i in graph.nodes
        ] for p in partitions
    ])
    print('Binary array created...')
    if set(A.ravel()) != set([0,1]):
        print('binary matrix contains invalid entries:', set(A.ravel()))
        print([(i,j,A[i,j]) for i in range(len(A)) for j in range(len(A)) if A[i,j] not in [0,1]])
    AAT = A@A.T
    print('A@A.T has been computed...')
    match1 = 2*AAT + np.shape(A)[1] - np.sum(A, axis=1) - np.sum(A, axis=1).reshape(A.shape[0],1)
    match2 = np.shape(A)[1] - match1

    return np.minimum(match1, match2), np.argmin([match1, match2], axis=0)


def district_heat_map(gdf, partitions, matches, matchID=False, verbose=False, flip=False):
    # plots the location of the boundaries of partitions as a heat map
    graph = partitions[0].graph
    boundary_frequency = np.zeros(len(graph)) #ordered by graph
    A = np.array([
        [
            np.abs(matches[j] - p.assignment[i]) for i in graph.nodes
        ] for j, p in enumerate(partitions)
    ])
    gdf['d'] = np.mean(A, axis=0)
    if flip:
        gdf['d'] = 1-gdf['d']
    return gdf


def visualize_comparison2(partitions, coords, r, split_point=1000, 
                         hard=False, boundary=True, matches=None, enacted=True,
                         labels=['blocks', 'block groups']
    ):
    '''
    coords are circular coordinates in the interval [0,1]
    r is the radius
    '''
    
    #scatter plot on circle
    arcs = [
        [i for i, x in enumerate(coords) if n/8.0 <= x < (n+1)/8.0] for n in range(8)
    ]
    mycmap = plt.get_cmap('tab20')
    fig1, ax1 = plt.subplots(figsize=(5,5))
    for i, arc in enumerate(arcs):  
        ax1.scatter(
            [r*0.8*np.cos(2*np.pi*coords[x]) for x in arc if x < split_point],
            [r*0.8*np.sin(2*np.pi*coords[x]) for x in arc if x < split_point],
            c='tab:blue', #[mycmap(i)],
            s=15, zorder=1, marker='x'
        )
    ax1.set_ylim(-1.2*r, 1.2*r)
    ax1.set_xlim(-1.2*r, 1.2*r)
    ax1.set_aspect(1)
    ax1.axis('off')
    
    #1D histograms
    fig2, ax3 = plt.subplots(figsize=(8,6))
    ax3.hist(
        coords[:split_point],
        color='tab:blue',
        label=labels[0],
        alpha=0.5,
        bins=[x/50 for x in range(0,52)],
        zorder=1
    )
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_xticks([x/10 for x in range(11)])
    
    #boundary heatmaps
    print('Plotting heat maps...')
    previous_center_map = 0
    flip = True
    for i, arc in tqdm(enumerate(arcs)):
        #print(arc)
        if len(arc) > 0:
            ax2 = ax1.inset_axes(
                [0.3+1.2*np.cos(2*np.pi*(i/8.0+1/16))/2,
                 0.3+1.2*np.sin(2*np.pi*(i/8.0+1/16))/2,
                 .4,.4], zorder=-100
            )
            place = ax3.transAxes.inverted().transform(ax3.transData.transform((i/8+1/16,0)))[0]
            ax4 = ax3.inset_axes(
                [place-0.06, 0.7,.12,.12], zorder=-100
            )
            center_map = arc[np.argmin([coords[i] for i in arc])]
            flip = (flip == bool(matches[previous_center_map, center_map]))
            b = district_heat_map(
                gdf,
                [partitions[j] for j in arc],
                [matches[center_map, j] for j in arc],
                flip=flip
            )
            previous_center_map = center_map
            b['log_boundary'] = np.array(b['d'])
            b.plot(
                column='log_boundary',
                cmap='PiYG', ax=ax2
            )
            b.plot(
                column='log_boundary',
                cmap='PiYG', ax=ax4
            )
        ax2.axis('off')
        ax4.axis('off')
        
    return fig1, fig2

def load_gdf(statename, level):
    if level == 'blockgroups':
        if statename == 'ID':
            gdf = gpd.read_file(
                'blockgroups/nhgis0024_shape/nhgis0024_shapefile_tl2020_160_blck_grp_2020/ID_blck_grp_2020.shp'
            )
        if statename == 'ME':
            gdf = gpd.read_file(
                'blockgroups/nhgis0024_shape/nhgis0024_shapefile_tl2020_230_blck_grp_2020/ME_blck_grp_2020.shp'
            )
        if statename == 'NH':
            gdf = gpd.read_file(
                'blockgroups/nhgis0024_shape/nhgis0024_shapefile_tl2020_330_blck_grp_2020/NH_blck_grp_2020.shp'
            )
        if statename == 'RI':
            gdf = gpd.read_file(
                'blockgroups/nhgis0024_shape/nhgis0024_shapefile_tl2020_440_blck_grp_2020/RI_blck_grp_2020.shp'
            )
        if statename == 'MT':
            gdf = gpd.read_file(
                'blockgroups/nhgis0024_shape/nhgis0024_shapefile_tl2020_300_blck_grp_2020/MT_blck_grp_2020.shp'
            )
        if statename == 'WV':
            gdf = gpd.read_file(
                'blockgroups/nhgis0024_shape/nhgis0024_shapefile_tl2020_540_blck_grp_2020/WV_blck_grp_2020.shp'
            )
        df = pd.read_csv('blockgroups/nhgis0024_csv/nhgis0024_ds248_2020_blck_grp.csv', index_col='GISJOIN')
        gdf = gdf.join(df, on='GISJOIN', rsuffix='_y')
        gdf = gdf.rename(columns={'U7B001':'TOTPOP'})
        election_updaters = {}
        current_col = None

    if level == 'blocks':
        print('Census blocks not supported in this version.')
        return None

    return gdf

for statename in ['ME']: #full list is ['ID', 'MT', 'ME', 'WV', 'NH', 'RI']
    print(statename, '\n-------\n')

    #get the prestored partitions
    blockgroup_assignments = pickle.load(open('ensembles/assignment_{}_{}_{}.p'.format(statename, 'blockgroups', tag), 'rb'))
    blockgroup_graph = Graph.from_json('ensembles/graph_{}_{}.p'.format(statename, 'blockgroups'))
    bg_partitions = [Partition(blockgroup_graph, a) for a in blockgroup_assignments]
    partitions = bg_partitions

    #get the gdf and trim disconnected pieces
    gdf = load_gdf(statename, 'blockgroups')
    nodes = frozenset(blockgroup_graph.nodes)
    gdf = gdf.drop(labels=[x for x in gdf.index if x not in nodes], axis='index')  

    #get pairwise distances
    M, matches = pairwise_hamming(partitions)
    assert np.min(M) >= 0
    assert np.array_equal(M, M.T)

    #embed into circle
    np.random.seed(2023) 
    r, y, losses = fit_to_circle_ot_start(M, n_points=1000)

    #make figures
    fig1, fig2 = visualize_comparison2(
        partitions, y, r, split_point=1000,
        matches=matches, boundary=False, enacted=False
    )
    fig1.savefig('redistricting_figures/circle_{}_{}_{}.png'.format(statename, level, tag), bbox_inches='tight', dpi=300)
    fig2.savefig('redistricting_figures/hist_{}_{}_{}.png'.format(statename, level, tag), bbox_inches='tight', dpi=300)






