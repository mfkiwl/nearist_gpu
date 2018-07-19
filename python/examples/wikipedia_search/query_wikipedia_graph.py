"""
Purpose:

This code returns the titles of the 10 most similar Wikipedia articles against 
a given query wikipedia article title.

Suggested use:

Copy code into an IDE that keeps variables in memory (Spyder, Jupyter, etc.). This way,
the datasets in memory and connection to Nearist server will persist while you 
iteratively search new query terms and adjust parameters.

Query term: 

The name of a wikipedia article title. Typically only the first word of the
title is capitalized, unless the title is someone's name, etc. Refer to 
wikipedia online to get the correct title.
"""

import h5py
import pickle
import time

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

# Filepath parameters 
path_on_local_drive = "./wiki_data/lsi_10NN_graph.h5"
path_to_titles = "./wiki_data/titles_to_id.pickle"

# ==============================
#       Load the dataset
# ==============================    
local_loaded = False
if not local_loaded:    
    print('Loading article titles...')
    
    # Load article titles
    title_to_idx = pickle.load( open( path_to_titles, "rb" ) )
    id_to_title = dict(zip(title_to_idx.values(), title_to_idx.keys()))
    
    # Open the local copy of the dataset file.
    h5f = h5py.File(path_on_local_drive, 'r')

# ==================================
#       Look up the results
# ==================================
print('\nFinding most similar articles to "%s"...\n' % query_article)
    
# Verify the specified article title is valid.
if not query_article in title_to_idx:        
    raise('Article title "%s" does not match any in the dataset.' % query_article)

t0 = time.time()

# Look up the index of the vector for this article.
query_idx = title_to_idx[query_article]

# Look up the nearest neighbors.
idxs = h5f['idxs'][query_idx, :]
dists = h5f['dists'][query_idx, :] 

# Print the article titles of the ten nearest neighbors.
# For each of the k results (omitting the top match, which
# is the query article itself)...
print('    %50s    Distance' % 'Title')
print('    %50s    ========' % '=====')
for i in range(1, len(idxs)):
    print('    %50s    %.3f' % ((id_to_title[idxs[i]]), dists[i]))

print('\nknn graph lookup took %.0f ms.' % ((time.time() - t0) * 1000.0))