from common import Request, Response, Status, Command
import sys
import socket
import time
import struct
import numpy as np
import binascii

class GpuClient:
    """
    This class provides the Python interface for communicating with the Nearist appliances.

    Commands are communicated via TCP/IP to the server.
    """

    def __init__(self):
        
        self.sock = None
        
        # These variables hold the elapsed time of the previous action.
        self.server_elapsed = 0
        self.client_elapsed = 0

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Called on leaving the 'with' block, whether or not there is an 
        exception. Ensures that the connection is closed properly.
        """
        # Close the connection.
        self.close()
        
        # Return False if there is an exception to re-raise it.
        if traceback:
            return False
        # Return True if there was no exception.
        else:
            return True

    @staticmethod
    def __recvall(sock, length):
        """
        Helper function to receive 'length' bytes or return None if EOF is hit.
        """
         
        data = bytearray()

        # Loop until we've received 'length' bytes.        
        while len(data) < length:

            # Receive the remaining bytes. The amount of data returned might
            # be less than 'length'.
            packet = sock.recv(length - len(data))

            # If we received 0 bytes, the connection has been closed...            
            if not packet:
                return None

            # Append the received bytes to 'data'.    
            data += packet

        # Return the 'length' bytes of received data.
        return data


    def __request(self, request):
        """
        Helper function to send a request to the server and receive the 
        response.
        
        :type request: Common.Request
        :param request: The Request object specifying the command to be sent
                        to the server.
        
        :rtype: Common.Response
        :return: The response object with the response body and decoded header.
        """
        
        # Format the packet and send it to the server.
        # 'sendall' will send all bytes in the request before returning.        
        status = self.sock.sendall(request.pack())

        # Check for status ('sendall' returns None if successful.)
        if status is not None:
            raise IOError("Transmission failure: %s" % str(status))
        
        # Receive the response header.
        # This call will block until the server has finished processing the
        # request and has sent a response.
        resp = Response()
        buf = GpuClient.__recvall(self.sock, resp.header_size)
        
        # Verify the buffer was received.
        if buf is None:
            raise IOError("Received 0 bytes from server, connection closed.")

        # Unpack the response header.
        resp.unpack_header(buf)

        # Check for bad status.            
        if resp.status != Status.SUCCESS:
            # Close the connection.
            self.close()
            
            # Raise the error received.
            raise IOError("Nearist error: %s " % Status(resp.status))
        
        # If the response includes a payload, receive it.
        if resp.body_length > 0:
            # Receive the body of this response.
            resp.body = GpuClient.__recvall(self.sock, resp.body_length)

            # Receive the body checksum
            checksum = GpuClient.__recvall(self.sock, 4)

            # Verify the checksum.
            if not checksum == struct.pack("=L", binascii.crc32(resp.body) & 0xFFFFFFFF):
                raise IOError("Response payload does not match checksum!")
            
        # Return the Response object.
        return resp

    def open(self, host, port, api_key):
        """
        Open a socket for communication with the Nearist appliance.
        
        :type host: string
        :param host: IP address of the Nearist server.
        
        :type port: integer
        :param port: Port number for accessing the Nearist server.
        
        :type api_key: string
        :param api_key: Unique user access key which is required to access the
                        server.

        """

        # Convert the host name and port to a 5-tuple of arguments.
        # We just need the "address family" parameter.
        address_info = socket.getaddrinfo(host, port)

        # Create a new socket (the host and port are specified in 'connect').
        self.sock = socket.socket(address_info[0][0], socket.SOCK_STREAM)

        # Connect to the host.
        self.sock.connect((host, port))

        # Store the API key        
        self.api_key = api_key
        
        return self

    def close(self):
        """
        Close the socket to the Nearist appliance.
        """
        self.sock.close()
            
    def load_dataset_file(self, file_name, dataset_name='', metric='L2'):
        """
        Load dataset which is already on the Nearist server hard disk.
        
        Datasets are uploaded to the Nearist server using SFTP, and then can be
        loaded into GPU memory remotely by this API.
        
        Both numpy and HDF5 file formats are supported. Numpy files must end
        in '.npy' and HDF5 files must end in '.h5'. The 'dataset_name' only 
        applies to HDF5 files.
        
        The metric choices are 'L2' or 'IP' for inner product. L2 is actually
        the squared L2 distance (which yields the same k-NN ranking as full
        L2). The inner product is used for cosine similarity; all vectors
        (dataset and query vectors) should be normalized first.
        
        :type file_name: string
        :param file_name: Path to the dataset file on the Nearist server.
        
        :type dataset_name: string
        :param dataset_name: Dataset name if file is HDF5 format.

        :type metric: string
        :param metric: 'L2' distance or 'IP' inner product similarity.

        """

        # Record the start time.
        t0 = time.time()
        
        # Construct a load request.
        req = Request(
            api_key = self.api_key,
            command = Command.LOAD_DATASET_FILE,
        )

        # Send the filename and dataset name as the packet payload.
        req.pack_json({
            "fileName": file_name, 
            "datasetName": dataset_name,
            "metric": metric
            
        })
        
        # Submit the request (__request handles the response status).
        resp = self.__request(req)
        
        # Record the elapsed server time, and the elapsed time from the 
        # client's perspective.
        self.server_elapsed = resp.elapsed
        self.client_elapsed = time.time() - t0


    def query(self, vectors, k=10, batch_size=128, verbose=False):
        """
        Submit a k-nearest neighbor search to the server.
        
        The maximum number of neighbors supported by the GPU is 1,024.
        
        The 'batch_size' and 'verbose' parameters are intended to help with
        experiments where you have a very large batch of input queries and need
        to break it into chunks, such as when running a pre-defined benchmark 
        like MNIST classification (10,000 query vectors).
        The 'batch_size' parameter will break the query set into smaller 
        batches for you, and print progress updates if 'verbose' is true.
        
        :type vectors: numpy.ndarray
        :param vectors: Matrix of query vectors, one per row. 

        :type k: int
        :param k: The number of nearest neighbors to find.
        
        :type batch_size: int
        :param batch_size: Sub-divide the 'vectors' into smaller batches in
                           order to receive progress updates.
        
        :type verbose: bool
        :param verbose: Print progress updates for queries consisting of 
                        multiple batches.
                        
        :returns: (distances, indeces). These are both matrices with shape 
                  [num queries x k]. That is, one row per query vector and one
                  column per nearest neighbor.
        """
        
        # Record the start time.
        t0 = time.time()
        
        # Validate the type of the vectors object.
        if not type(vectors) == np.ndarray:
            raise IOError("Query vectors should be of type numpy.ndarray")
        
        # Ensure vectors are type float32 as the server expects.
        if not vectors.dtype == np.float32:
            print("WARNING - Vectors are type %s but should be float32." \
                  " Casting vectors to float32." % str(vectors.dtype))
            # Cast the vectors float32 if they aren't already.      
            vectors = vectors.astype('float32')
        
        # Reset the elapsed time measurements.
        self.server_elapsed = 0
        self.client_elapsed = 0       
        
        # =================================
        #       Handle single queries 
        # =================================
        
        # If 'vectors' is just a single query vector...
        if vectors.ndim == 1:
            # Construct the query request.
            req = Request(
                api_key = self.api_key,
                command = Command.QUERY,
                k = k
            )
            
            # Add the vector.
            req.pack_vectors(vectors)
        
            # Submit the query and wait for the results.
            resp = self.__request(req)
            
            # Unpack the results and return them.
            D, I = resp.unpack_results()

            # Record the time spent on the server and the total time observed
            # by the client.
            self.server_elapsed = resp.elapsed
            self.client_elapsed = time.time() - t0

            return D, I
        
        # =================================
        #       Handle multiple queries 
        # =================================
        elif vectors.ndim == 2:
            # Transmit the queries in mini-batches in order to avoid memory errors
            # and to get progress updates.
            
            # Record the total number of query vectors.                       
            num_vecs = vectors.shape[0]

            # Verify there's at least one vector.
            if num_vecs == 0:
                raise IOError("Number of query vectors cannot be zero!")
            
            D_all = np.zeros((0, k))
            I_all = np.zeros((0, k), dtype='int32')
            
            start = 0
            
            # For each mini-batch...
            while start < num_vecs:
                # Calculate the 'end' of this mini-batch.
                end = min(start + batch_size, num_vecs)
    
                # Select the vectors in this mini-batch.
                mini_batch = vectors[start:end, :]
    
                # Progress update.
                if verbose and not start == 0:
                    # Caclulate the average throughput so far.
                    queries_per_sec = ((time.time() - t0)  / start)
                    
                    # Estimate how much time (in seconds) is left to complete the 
                    # test.
                    time_est = queries_per_sec * (len(vectors) - start)
                    
                    # Format the estimated time remaining into minutes.
                    if time_est < 90:
                        time_est_str = '~%.0f sec...' % time_est
                    else:
                        time_est_str = '~%.0f min...' % (time_est / 60.0)
    
                    print('  Query %5d / %5d (%3.0f%%) Time Remaining: %s' % (start, len(vectors), float(start) / len(vectors) * 100.0, time_est_str))
                    sys.stdout.flush()
                
                # Construct the query request.
                req = Request(
                    api_key = self.api_key,
                    command = Command.QUERY,
                    k = k
                )
                
                # Add the vectors.
                req.pack_vectors(mini_batch)
            
                # Submit the query and wait for the results.
                resp = self.__request(req)
    
                # Accumulate the total time spent on the server.
                self.server_elapsed += resp.elapsed
                
                # Unpack the results and return them.
                D, I = resp.unpack_results()
                
                if not len(mini_batch) == I.shape[0]:
                    print('ERROR: Mini batch length %d does not match results legnth %d!' % (len(mini_batch), I.shape[0]))
                    sys.stdout.flush()
                
                # Accumulate the results.
                D_all = np.concatenate((D_all, D))
                I_all = np.concatenate((I_all, I))
                
                # Update the start pointer.
                start = end
    
            # Store the total elapsed time from the client perspective.
            self.client_elapsed = time.time() - t0
            
            return D_all, I_all
        
        # If vectors.ndim is not 1 or 2, then somethings wrong with it.
        else:
            raise IOError("'vectors' argument has wrong number of dimensions!")
    
    def query_from_file(self, file_name, dataset_name='', k=10, batch_size=1024):
        """
        Perform a batch query using query vectors stored in a file on the 
        server.
        
        This can be used, for instance, to compute a k-nn graph on a dataset
        (i.e., find the 10 nearest neighbors for all vectors in a dataset). 
        
        Both numpy and HDF5 file formats are supported. Numpy files must end
        in '.npy' and HDF5 files must end in '.h5'. The 'dataset_name' only 
        applies to HDF5 files.
        
        The server will break the query vectors into batches of 'batch_size' to
        avoid memory errors.
        
        In addition to returning the results over the network, the server will
        also write them to an HDF5 file with the same location and name as the
        input file, with the suffix '*_results.h5'. This file contains two 
        datasets named 'distances' and 'indeces' which are the results of the 
        search. This file can be retrieved from the server in the event of a
        failed connection during processing.
        
        :type file_name: string
        :param file_name: Path to the query vectors file on the Nearist server.
        
        :type dataset_name: string
        :param dataset_name: Dataset name if file is HDF5 format.

        :type k: int
        :param k: Number of neighbors to return for each query.
        
        :type batch_size: int
        :param batch_size: Number of query vectors to submit at a time. 

                
        """
        # Record the start time.
        t0 = time.time()
        
        # Construct a load request.
        req = Request(
            api_key = self.api_key,
            command = Command.QUERY_FROM_FILE,
        )

        # Send the filename and dataset name as the packet payload.
        req.pack_json({"fileName": file_name, "datasetName": dataset_name, "k": k, "batchSize": batch_size})
        
        # Submit the request (__request handles the response status).
        resp = self.__request(req)

        # Unpack the results.
        D, I = resp.unpack_results()
        
        # Record the elapsed server time, and the elapsed time from the 
        # client's perspective.
        self.server_elapsed = resp.elapsed
        self.client_elapsed = time.time() - t0
        
        return D, I
    
    def print_timings(self):
        """
        Print the timing measurements from the previous command.
        
        - 'Server Time' is the total time spent on the Nearist server.
        - 'Overhead' is the difference between the Client's observed time
        and the server time, which is caused by network communication overhead.
        - 'Total' is the total time elapsed from the Client's perspective.
        """
        print("\nTiming breakdown:")
        
        if self.server_elapsed < 1.0:
            print("  Server Time: %0.0f ms" % (self.server_elapsed * 1000))
            print("     Overhead: %0.0f ms" % ((self.client_elapsed - self.server_elapsed) * 1000))
            print("        Total: %0.0f ms" % (self.client_elapsed * 1000))
        elif self.server_elapsed < 120.0:
            print("  Server Time: %0.1f sec" % self.server_elapsed)
            print("     Overhead: %0.1f sec" % (self.client_elapsed - self.server_elapsed))
            print("        Total: %0.1f sec" % self.client_elapsed)
        else:
            print("  Server Time: %0.1f min" % (self.server_elapsed / 60.0))
            print("     Overhead: %0.1f min" % ((self.client_elapsed - self.server_elapsed) / 60.0))
            print("        Total: %0.1f min" % (self.client_elapsed / 60.0))
