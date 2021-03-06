# Wikipedia Article Similarity Search

In this example, you specify the title of an article on Wikipedia, and the hardware performs a k-NN search to retrieve the ten most conceptually similar articles on Wikipedia. 

We have applied Latent Semantic Indexing (LSI) to create vector representations for the roughly 4.2M articles on Wikipedia. We used the [wiki-sim-search](http://github.com/chrisjmccormick/wiki-sim-search/) project available on GitHub to generate this dataset.

This example includes three components:
1. `query_wikipedia_live.py` - This script uses the GpuClient to perform a single search.
2. `wikipedia_knn_graph.py` - This script finds the 10 nearest neighbors for all vectors in the dataset.
3. `query_wikipedia_graph.py` - This script uses the precomputed graph from the previous script to perform the search.

## Pre-Requisites

In order to run these examples, you will need the following dataset files downloaded to your local machine. The scripts expect them to be found in a subdirectory named `wiki_data`
<table>
  <tr>  <th>File</th>               <th>Size</th>    <th>Description</th>   </tr>
 
  <tr>  <td><a href="https://drive.google.com/a/nearist.ai/file/d/1dNUxJVZSc9oQxuQp2ot1W2m4mdU-9qXl/view?usp=sharing">lsi_index_float32.h5</a></td>   <td>4.7 GB</td>   <td>LSI vectors for every article in Wikipedia. ~4.2 million rows with 300 features each.</td>    </tr>
  
  <tr>  <td><a href="https://drive.google.com/open?id=1wCg61RgNc0LbMjePSZUBUIxg1T46FujY">titles_to_id.pickle</a></td>   <td>130 MB</td>  <td>Dictionary for looking up the ID of an article with a given title.
</td>    </tr>  
</table>

You'll also need:
* The Nearist classes ('/nearist_gpu/python/src/') on your Python Path.
* The API access key generated for your user account.
* An active rental session and the IP address of your reserved Nearist server.
    * _Make sure to update your copy of the example code with your API key and the provided IP address before running._

## Running the examples

You may run the script within your favorite Python IDE, or, run from a Python terminal with the command ```exec(open("query_wikipedia_live.py").read(), globals())```

### Load once, run multiple times
Loading the dataset, both locally and on the remote server, is somewhat time consuming. Fortunately, this step doesn't need to be performed on every run of the scripts. After your first run of the example, look for the 'loaded' flag(s) in the script and set these to true, and it will skip the load steps for subsequent runs.

### Expected Output - query_wikipedia_live
The following output was generated on an Amazon P2 instance with a single Tesla K80. 

```
Connecting to Nearist server...
    Connection successful.

Loading dataset vectors from remote server into GPU...
Loading article titles...
Retrieving local query vector...

Finding most similar articles to "Water treatment"...

                                                 Title    Distance
                                                 =====    ========
                            Industrial water treatment    0.025
                                        Drinking water    0.042
                                             Raw water    0.051
                                       Reclaimed water    0.051
                                    Water chlorination    0.056
                                          Water bottle    0.056
                                    Water purification    0.057
                                 Groundwater pollution    0.060
                                        Purified water    0.064
                  Water issues in developing countries    0.064

Timing breakdown:
  Server Time: 83 ms
     Overhead: 101 ms
        Total: 184 ms
```

The server time with a Tesla V100 is about 22ms (almost 4x faster!).

### Expected Output - wikipedia_knn_graph.py
The following output was generated on an Amazon P3 instance with a single Tesla V100 GPU.

```
Connecting to Nearist server...
    Connection successful.

Loading dataset vectors from remote server into GPU...
Performing a test query to estimate throughput...
GPU throughput (w/ batch size of 1024) is 4504 queries per second.
Estimated time to complete the knn graph: 16 min

Running knn-table for 4198780 vectors, batch_size 1024, with k=11...

Timing breakdown:
  Server Time: 20.6 min
     Overhead: 1.2 min
        Total: 21.8 min

Writing results to local disk...
Done.
```

### Expected Output - query_wikipedia_graph.py
Looking up the results in the pre-computed knn graph is nearly instantaneous.
```
Finding most similar articles to "Water treatment"...

                                                 Title    Distance
                                                 =====    ========
                            Industrial water treatment    0.025
                                        Drinking water    0.042
                                             Raw water    0.051
                                       Reclaimed water    0.051
                                    Water chlorination    0.056
                                          Water bottle    0.056
                                    Water purification    0.057
                                 Groundwater pollution    0.060
                                        Purified water    0.064
                  Water issues in developing countries    0.064

knn graph lookup took 2 ms.
```
