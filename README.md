# Generalized Dimension Reduction Using Semi-Relaxed Gromov-Wasserstein Distance

This code produces the results and figures in the paper _Generalized Dimension Reduction Using Semi-Relaxed Gromov-Wasserstein Distance_ by Ranthony A. Clark, Tom Needham and Thomas Weighill.

## Requirements

In addition to the requirements in `requirements.txt`, the code also requires [this](https://github.com/TJPaik/CircularCoordinates) repo, cloned into a folder at the same level as this repo's enclosing folder. It also requires downloading the MNIST data set as a `.pkl.gz` file from [here](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz). For the redistricting experiments, a block group shapefile is provided in this folder for Maine only due to size restrictions. For other shapefiles, please download the block group shapefile for the rest of the states from `https://www.nhgis.org/`.

## Scripts 

The following scripts generate the images in the paper

- `run-cities-embedding.py`: generates the Cities dataset embeddings and figures.
- `run-MNIST-embeddings-unrotated.py`: generates the MNIST dataset embeddings and figures
- `run-MNIST-embeddings.py`: generates the rotated MNIST dataset embeddings and figures
- `run-redistricting-embeddings.py`: generates the redistricting embeddings figures.

Before you run `run-redistricting-embeddings.py`, you need to run `generate-bipartitions.py` for each state to generate the ensembles. 
