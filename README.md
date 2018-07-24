

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

