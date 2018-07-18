from Common import *
import sys
import json
import socket
import time
import numpy as np

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
        buf = Client.__recvall(self.sock, resp.header_size)
        
        # Verify the buffer was received.
        if buf is None:
            raise IOError("Received 0 bytes from server, connection closed.")

        # Unpack the response header.
        resp.unpack_header(buf)

        # Check for bad status.            
        if resp.status != Status.SUCCESS:
            raise IOError("Nearist error: %s " % Status(resp.status))
        
        # If the response includes a payload, receive it.
        if resp.body_length > 0:
            # Receive the body of this response.
            resp.body = Client.__recvall(self.sock, resp.body_length)

            # Receive the body checksum
            checksum = Client.__recvall(self.sock, 4)

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

    def close(self):
        """
        Close the socket to the Nearist appliance.
        """
        self.sock.close()
            
    def load_dataset_file(self, file_name, dataset_name):
        """
        Load dataset which is already on the Nearist server hard disk.
        
        Datasets are uploaded to the Nearist server using FTP, and then can be
        loaded into GPU memory 
        
        :type file_name: string
        :param file_name: Path to the dataset file on the Nearist server.
        
        :type dataset_name: string
        :param dataset_name: Local dataset name

        """

        # Record the start time.
        t0 = time.time()
        
        # Construct a load request.
        req = Request(
            api_key = self.api_key,
            command = Command.LOAD_DATASET_FILE,
        )

        # Send the filename and dataset name as the packet payload.
        req.pack_json({"fileName": file_name, "datasetName": dataset_name})
        
        # Submit the request (__request handles the response status).
        resp = self.__request(req)
        
        # Record the elapsed server time, and the elapsed time from the 
        # client's perspective.
        self.server_elapsed = resp.elapsed
        self.client_elapsed = time.time() - t0


    def query(self, vectors, k=10, batch_size=128, verbose=False):
        """
        Submit a k-nearest neighbor search to the server.
        
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
        
        start = 0
        results = []
        
        # Record the start time.
        t0 = time.time()
        
        # Reset the elapsed time measurements.
        self.server_elapsed = 0
        self.client_elapsed = 0
        
        # Transmit the queries in mini-batches in order to avoid memory errors
        # and to get progress updates.
        
        D_all = np.zeros((0, k))
        I_all = np.zeros((0, k), dtype='int32')
        
        # For each mini-batch...
        while start < vectors.shape[0]:
            # Calculate the 'end' of this mini-batch.
            end = min(start + batch_size, len(vectors))

            # Select the vectors in this mini-batch.
            mini_batch = vectors[start:end]

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
        else:
            print("  Server Time: %0.1f sec" % self.server_elapsed)
            print("     Overhead: %0.1f sec" % (self.client_elapsed - self.server_elapsed))
            print("        Total: %0.1f sec" % self.client_elapsed)