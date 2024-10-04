from gerrychain import Graph, Election, updaters, Partition, constraints, MarkovChain
from gerrychain.updaters import cut_edges
from gerrychain.random import random
from gerrychain.proposals import recom
from gerrychain.tree import recursive_tree_part
from gerrychain.accept import always_accept
from gerrychain import random
import numpy as np
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

statename = 'MT' #put state abbreviation here

np.random.seed(2022)
random.seed(2022)
level = 'blockgroups'
pop_col = 'TOTPOP'
tag = 'vanilla'

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
        
else:
    print('Census blocks not supported in this version.')


print('Geodataframe loaded, creating graph now...')
gdf.geometry = gdf.geometry.buffer(0)
graph = Graph.from_geodataframe(gdf, ignore_errors=True)
graph.add_data(gdf)
print(nx.number_connected_components(graph), " components")
if nx.number_connected_components(graph) > 1:
    graph = graph.subgraph(
        max(nx.connected_components(graph), key=len)
    )
gdf = gdf.drop(labels=[x for x in gdf.index if x not in list(graph.nodes)], axis='index')
graph.to_json('ensembles/graph_{}_{}.p'.format(statename, level))


# parameters
tag = 'vanilla'
steps = 1000
INTERVAL = 1
num_districts = 2
pop_tol = 0.02
total_population = sum([graph.nodes[n][pop_col] for n in graph.nodes()])
k = num_districts
pop_target = total_population/k
myproposal = partial(recom, pop_col=pop_col, pop_target=pop_target, epsilon=pop_tol, node_repeats=10)

# updaters
myupdaters = {
    "population": updaters.Tally(pop_col, alias="population"),
    "cut_edges": cut_edges
}
myupdaters.update(election_updaters)

# create initial partition
print("Creating initial partition with", k, "districts...", end="")
initial_ass = recursive_tree_part(graph, range(k), pop_target, pop_col, pop_tol, node_repeats=10)
initial_partition = Partition(graph, initial_ass, myupdaters)

# set up chain
myconstraints = [
    constraints.within_percent_of_ideal_population(initial_partition, pop_tol),
]
chain = MarkovChain(
    proposal=myproposal,
    constraints=myconstraints,
    accept=always_accept,
    initial_state=initial_partition,
    total_steps=steps
)

# run ReCom
partitions = []
print('Running chain...')
for index, step in tqdm(enumerate(chain)):
    #store some plans
    if index%INTERVAL == 0:
        partitions.append(step)
        

pickle.dump([p.assignment for p in partitions], open('ensembles/assignment_{}_{}_{}.p'.format(statename, level, tag), 'wb'))
#assignments = pickle.load(open('ensembles/assignment_{}_{}_{}.p'.format(statename, level, tag), 'rb'))
#partitions = [Partition(graph, a, myupdaters) for a in assignments]




