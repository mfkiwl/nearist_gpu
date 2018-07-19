from __future__ import division
import h5py
import sys
import numpy as np
from gpuclient import GpuClient

# ==============================
#     Parameters
# ==============================

# Connection parameters
api_key = "apikey"
nearist_port = 0
nearist_ip = ""

# Filepath parameters 
path_on_nearist_server = "/nearist/Wikipedia/lsi_index_float32.h5"
path_on_local_drive = "./wiki_data/lsi_index_float32.h5"
output_on_local_drive = "./wiki_data/lsi_10NN_graph.h5"

# Number of queries for the GPU to process at once.
# The GPU is most efficient when the input batch is sufficiently large. 
# However, if the input batch is too large then there will be memory issues.
# A batch size of 1024 is good for the Wikipedia dataset.
batch_size = 1024

# Get the top 11 neighbors, since the first nearest neighbor is the query
# vector itself. We'll toss the first neighbor, leaving us with 10 neighbors.
k = 11

# ==============================
#     Establish connection
# ==============================

print('Connecting to Nearist server...')

c = GpuClient()
c.open(nearist_ip, nearist_port, api_key)

print('    Connection successful.\n')

# ==============================
#       Load the dataset
# ==============================
# Set the 'loaded' flag to 'True' after the first run of this script so that
# you don't have to load anything again.
loaded = False
if not loaded:
    print('Loading dataset vectors from remote server into GPU...')

    # Load dataset into GPU memory.
    c.load_dataset_file(file_name=path_on_nearist_server, dataset_name='lsi')

# ==============================
#      Measure Throughput
# ==============================

print('Performing a test query to estimate throughput...')

# Use a handful of batches to estimate throughput.
test_batch_size = 4 * batch_size

# Open the local copy of the dataset file.
h5f = h5py.File(path_on_local_drive, 'r')

# Read the first four batches of the Wikipedia vectors into local memory.
vectors = h5f['lsi'][0:test_batch_size, :]

# Perform a query on the first four batches.
dists, idxs = c.query(vectors, k=k, batch_size=batch_size)
    
# Measure the GPU throughput (queries per second) based on this test.
# We don't need to include internet overhead in this measurement because the
# query vectors for the knn table are already on the remote server.
throughput = test_batch_size / c.server_elapsed

# Estimate the time to complete the whole graph in minutes
est_graph_time = h5f['lsi'].shape[0] / throughput / 60.0

print('GPU throughput (w/ batch size of %d) is %.0f queries per second.' % (batch_size, throughput))
print('Estimated time to complete the knn graph: %.0f min\n' % est_graph_time)

# ==============================
#      Compute 10-NN graph
# ==============================
print('Running knn-table for %d vectors, batch_size %d, with k=%d...' % (h5f['lsi'].shape[0], batch_size, k))
sys.stdout.flush()

# Specify the dataset file on the Nearist GPU server to read the query vectors 
# from.
dists, idxs = c.query_from_file(file_name=path_on_nearist_server,
                                dataset_name='lsi',
                                k=k, batch_size=batch_size)

# Print out the measured timings.
c.print_timings()

# Close the server connection
c.close()

# ==============================
#      Store the results
# ==============================

# Delete column 0 from the results                                
np.delete(dists, 0, axis=1)
np.delete(idxs, 0, axis=1)

print('Writing results to local disk...')

# Write out the two matrices to an HDF5 file.
h5f = h5py.File(output_on_local_drive, 'w')
h5f.create_dataset(name='dists', data=dists)
h5f.create_dataset(name='idxs', data=idxs)
h5f.close()

print('Done')
