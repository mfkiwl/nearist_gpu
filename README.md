

![alt text](http://nearist.sightbox.io/wp-content/uploads/2017/04/nearist.svg)

## Structure
This repository contains Nearist's Client API for interacting with Nearist GPU-based servers remotely, along with example code demonstrating different uses and applications.

- `/python/src` contains the Client interface code for interacting with Nearist servers.
- `/python/examples` contains pre-made examples of different nearest neighbor search tasks.

Please visit the [wiki](https://github.com/nearist/nearist/wiki) for documentation, tutorials, and example applications.

## Preliminaries and Installation

You will have to obtain an API key in order to gain access to Nearist servers. 

To install, clone this repository and add the `/nearist_gpu/python/src/` directory to your `PYTHONPATH`.

To run any of the examples, you'll need to edit the example script to specify your API access key and supplied IP address.

## API: At a Glance

Below is the basic usage for connecting to the server, loading your data, and performing a k-NN search.

```python
 1: # Import the Python API
 2: from nearist import Client 
 3: c = Client()
 4: 
 5: # Open the connection to the server (IP, port, API key)
 6: c.open("103.210.163.290", 9885, api_key)
 7:
 8: # Load your dataset into memory on Nearist servers
 9: c.load_dataset_file(file_name='dataset.h5', dataset_name='vectors')
10:
11: # Load local vectors to be queried against the dataset
12: query_vectors = load(path='my_local_dataset.h5')[:10] 
13:
14: # Perform a 10-NN query for all query vectors.
15: distances, indeces = c.query(query_vectors, k=10)
```

## Example: MNIST

To show how Nearist works, we will run through the provided code in `/python/examples/MNIST/` to perform nearest neighbors classification of the MNIST dataset. 

You can download the generated dataset directly from [here](https://drive.google.com/drive/folders/1tr-q_uhg6PVuQKIwnLDRMtRsrG2oyS8C)). The dataset vectors were generated from a convolutional neural network to transform the raw MNIST images into 1024-dimensional vectors. 

You should now have the HDF5 (extension '.h5') datasets stored in your `data` folder.

After updating the script with the server IP address, port number, and your API key, run:

`python run_classification.py`

This will return the elapsed time and accuracy of the nearest neighbors classifier.
