

![alt text](http://nearist.sightbox.io/wp-content/uploads/2017/04/nearist.svg)

## Structure
This repository contains Nearist's GpuClient API for interacting with Nearist GPU-based servers remotely, along with example code demonstrating different uses and applications.

- `/python/src` contains the GpuClient interface code for interacting with Nearist servers.
- `/python/examples` contains pre-made examples of different nearest neighbor search tasks.

## Preliminaries and Installation

To install, clone this repository and add the `/nearist_gpu/python/src/` directory to your `PYTHONPATH`.

You will have to obtain an API key in order to gain access to Nearist servers. Once you are provided with a key and server IP address, fill in these values in the example scripts before running them.

## API: At a Glance

Below is the basic usage for connecting to the server, loading your data, and performing a k-NN search.

```python
 1: # Import the Python API
 2: from gpuclient import GpuClient 
 3: c = GpuClient()
 4: 
 5: # Open the connection to the server (IP, port, API key)
 6: with c.open("103.210.163.290", 9885, api_key):
 7:
 8:     # Load your dataset into memory on Nearist servers
 9:     c.load_dataset_file(file_name='dataset.h5', dataset_name='vectors', metric='L2')
10:
11:     # Load local vectors to be queried against the dataset
12:     query_vectors = load(path='my_local_dataset.h5')[:10] 
13:
14:     # Perform a 10-NN query for all query vectors.
15:     distances, indeces = c.query(query_vectors, k=10)
```

## Getting Started
To get started quickly with a pre-built example, check out the Wikipedia example and its README [here](https://github.com/nearist/nearist_gpu/tree/master/python/examples/wikipedia_search).

## API Overview

The basic commands are:
* `open(ip_address, port_number, api_key)` - Establish a connection to the Nearist GPU server.
    * `open` should always be called using a `with` statement to ensure the connection is properly closed when your code exits.
* `load_dataset_file(filename, dataset_name, metric)` - Load your dataset vectors, i.e., the vectors to be searched.
    * Vectors are stored on disk using either the Numpy (.npy) or HDF5 (.h5) file formats and transferred to the server using SFTP.
    * The supported metrics are 'L2' (squared L2 distance) or 'IP' (the inner-product, which yields cosine similarity if the vectors are all normalized).
* `distances, indexes = query(query_vectors, k)` - Submit one or more query vectors for k-nn search.
    * The number of neighbors returned, `k`, can be any value up to 1,024.

Additional documentation can be found in the detailed function header comments in [gpuclient.py](https://github.com/nearist/nearist_gpu/blob/master/python/src/gpuclient.py).

## Distance Metrics
The two supported distance metrics are 'L2' and 'IP'.

The 'L2' distance is technically the squared-L2 distance, meaning that we don't take the final square root step. However, the squared-L2 distance produces identical rankings in k-NN searches.

The 'IP' metric is the inner-product (also called the dot product). 

'IP' can be used to perform k-NN search with Cosine similarity by simply pre-normalizing all vectors. The equation for the cosine similarity between two vectors `a` and `b` is:
![Equation for cosine similarity](http://mccormickml.com/assets/cosine_l1/cosine_similarity.png)

The following code shows how to normalize a matrix of vectors `X` (containing one vector per row) using NumPy:

```python
# Calculate the L2-norms for all rows of X.
l2norms = np.linalg.norm(X, axis=1, ord=2)

# Reshape the norms into a column vector.
l2norms = l2norms.reshape(len(l2norms), 1)

# Divide all of the row vectors in X by their norms to normalize them.
X_norm = X / l2norms
```

Save the normalized version of your dataset and upload it to the GPU server. Then, normalize all query vectors before performing the search.

## Working with your own datasets
To upload your own dataset vectors to be searched, we provide SFTP (file transfer over SSH) access to the GPU server. 

We will provide you with a username, and will need to add your public key to the server's authorized list.

Your account will have limited functionality--SFTP access is enabled but not SSH or SCP. You will be logged into a 'datasets' folder, and are restricted to this directory and those beneath it. There is a `nearist` subdirectory containing pre-built example datasets from Nearist, and a `[company name]` subdirectory where you can upload your own dataset files.