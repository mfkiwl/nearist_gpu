"""
Purpose:

This code returns the titles of the 10 most similar Wikipedia articles against a 
given query wikipedia article title.



Suggested use:

Copy code into an IDE that keeps variables in memory (Spyder, Jupyter, etc.). This way,
the datasets in memory and connection to Nearist server will persist while you 
iteratively search new query terms and adjust parameters.



Query term: 

The name of a wikipedia article title. Titles are capitalized, and typically 
each word of a multiword title is capitalizaed. Refer to wikipedia article titles online.

"""

from gpuclient import GpuClient
import h5py
import pickle

# ==============================
#     Parameters
# ==============================

# Specify the title of the Wikipedia article to use as the search query.
#
# Multi-word titles can specified with spaces, such as "Abraham Lincoln".
# Capitalization should match the Wikipedia article title exactly. See the
# examples below.

#query_article = "Abraham Lincoln"
#query_article = "Computer science"
query_article = "Water treatment"

# Connection parameters
api_key = ""
nearist_port = 0
nearist_ip = ""

# Filepath parameters 
path_on_nearist_server = "/nearist/Wikipedia/lsi_index_float32.h5"
path_on_local_drive = "./wiki_data/lsi_index_float32.h5"
path_to_titles = "./wiki_data/titles_to_id.pickle"

# ==============================
#     Establish connection
# ==============================

print('Connecting to Nearist server...')

c = GpuClient()

# Always use the GpuClient inside a with statement--this ensures that the
# connection can be properly closed even if an exception is thrown.
with c.open(nearist_ip, nearist_port, api_key):

    print('    Connection successful.\n')
    
    # ==============================
    #       Load the dataset
    # ==============================
    # Set the '*_loaded' flags to 'True' after the first run of this script so 
    # that you don't have to load anything again.
    remote_loaded = False
    if not remote_loaded:
        print('Loading dataset vectors from remote server into GPU...')
    
        # Load dataset into GPU memory.
        c.load_dataset_file(
            file_name = path_on_nearist_server, # Remote file path
            dataset_name = 'lsi'  # Dataset name within HDF5 file
        )
        
    local_loaded = False
    if not local_loaded:    
        print('Loading article titles...')
        
        # Load dictionary mapping article titles to their vector index.
        title_to_idx = pickle.load( open( path_to_titles, "rb" ) )
        
        # Also create a map from index to title (for displaying the
        # search results).
        id_to_title = dict(zip(title_to_idx.values(), title_to_idx.keys()))
        
        # Open the *local* copy of the dataset file.
        h5f = h5py.File(path_on_local_drive, 'r')
    
    # ==================================
    #       Load the query vector
    # ==================================
    print('Retrieving local query vector...')
        
    # Verify the specified article title is valid.
    if not query_article in title_to_idx:        
        c.close()
        raise('Article title "%s" does not match any in the \
              dataset.' % query_article)
    
    # Look up the index of the vector for this article.
    query_idx = title_to_idx[query_article]
    
    # Read in the query vector.
    query_vec = h5f['lsi'][query_idx].astype('float32')
    
    # =======================================
    #       Search for similar articles
    # =======================================
    print('\nFinding most similar articles to "%s"...\n' % query_article)
    
    # Set k = 11 to get 10 results since the top result will always be the 
    # query article itself.
    k = 11
    
    # Submit the query to the Nearist GPU server.    
    dists, idxs = c.query(query_vec, k=k)
    
    # Print the article titles of the ten nearest neighbors.
    # For each of the k results (omitting the top match, which
    # is the query article itself)...
    print('    %50s    Distance' % 'Title')
    print('    %50s    ========' % '=====')
    for i in range(1, k):
        print('    %50s    %.3f' % ((id_to_title[idxs[0, i]]), dists[0, i]))
    
    # Print out the timing measurements for the query operation.
    c.print_timings()
    
    # Close the server connection (this is redundant to 'with', but fine).
    c.close()