import struct
import binascii
import json
import pickle
import numpy as np
from enum import IntEnum


class Command(IntEnum):
    """
    Command IDs.
    """
    LOAD_DATASET_FILE = 0x01
    QUERY = 0x02
    QUERY_FROM_FILE = 0x03


class Status(IntEnum):
    """
    
    """
    SUCCESS = 0x00
    INVALID_SEQUENCE = 0x01
    INVALID_ARGUMENT = 0x02
    INVALID_PACKET = 0x03
    NOT_SUPPORTED = 0x04
    INVALID_COMMAND = 0x05
    INVALID_DATA = 0x06
    TIMEOUT = 0x07
    INVALID_CHECKSUM = 0x08
    INVALID_API_KEY = 0x09
    DIFFERENT_VECTOR_LENGTH = 0x10 # When query vectors don't match dataset
    DATASET_FILE_NOT_FOUND = 0x20 #
    DATASET_NOT_FOUND = 0x21
    DATASET_SIZE_NOT_SUPPORTED = 0x22
    QUERY_SIZE_NOT_SUPPORTED = 0x23
    DISTANCE_MODE_NOT_SUPPORTED = 0x24
    QUERY_MODE_NOT_SUPPORTED = 0x25
    READ_COUNT_NOT_SUPPORTED = 0x26
    UNKNOWN_ERROR = 0xFF


class Request:
    """
    Class representing a request received from the client.
    
    The request header is 28 bytes:
        (4) command
        (4) k
        (8) API key
        (8) body_length
        (4) header checksum
    
    The body is received over the socket separately from the header.
    First we receive 28 bytes to receive the header, then this tells us
    how many bytes to receive for the body.
    """
    def __init__(self, api_key='', command=None, k=0, body_length=0, body=None):
        """
        The sender will call this constructor and pass in the arguments.
        The receiver will create the instance with default values, and then unpack the buffer.
        """

        self.command = command
        self.k = k
        
        # Pad the API key out to 8 characters, and only take 8 characters.
        self.api_key = api_key.ljust(8)[0:8]
        
        self.body_length = body_length
        self.body = body
        
        # Store the header size to be referenced elsewhere in the code.
        self.header_size = 28

    def pack_json(self, obj):
        """
        Formats the JSON object into a string to be included in the packet.
        """
        # Use the json module to construct a string representation and add it
        # to the request.
        self.body = json.dumps(obj)
        self.body_length = len(self.body)
    
    def unpack_json(self):
        """
        Parses json string into an object.
        """
        return json.loads(self.body)
    
    def pack_vectors(self, vectors):
        """
        Converts the matrix of vectors into bytes as a packet payload.
        """
        # Numpy ndarray has a function specifically for conversion to a byte 
        # string.
        self.body = vectors.tobytes()
        
        # Store the length in bytes of the body.
        self.body_length = len(self.body)
    
    def unpack_vectors(self, dim):
        """
        Parse the vectors from the packet body.
        
        The body length is compared against the expected vector length before
        this function is called.
        
        :type dim: int
        :param dim: Length of vectors (number of components, not bytes).
        
        :rtype: numpy.ndarray
        :return: Matrix of vectors, one per row.
        """
        # Numpy implements a function for converting from bytes to ndarray.
        vectors = np.frombuffer(self.body, dtype='float32')
                
        # Infer the number of vectors in the payload based on the vector 
        # length.
        num_vecs = int(len(vectors) / dim)
        
        # Reshape the array into a matrix of vectors.
        return vectors.reshape((num_vecs, dim))   
    
    def pack(self):
        """
        Returns the binary representation of this request (as a string).
        
        This function should only be called once all properties of the request
        have been set (including any payload).
        
        The request header is 28 bytes:
            (4) command
            (4) k
            (8) API key
            (8) body_length
            (4) header checksum  
            
        The header is followed by the body of the request, if present.
        
        :rtype: string
        :return: Completed packet encoded in bytes.
        """
        
        # Pack the message header.        
        #   'L' is unsigned long (32-bit, 4 bytes)
        
        buf = struct.pack("=LL", self.command, self.k)

        # Add the API key to the header (it's 8 characters, 8 bytes).
        buf += self.api_key
        
        # 'Q' is unsigned long long (64-bit, 8 bytes)
        buf += struct.pack("=Q", self.body_length)
        
        # Add a checksum of the header fields.
        buf += struct.pack("=L", binascii.crc32(buf) & 0xFFFFFFFF)

        # If this request includes a body...
        if self.body_length > 0 and self.body is not None:
            
            # Add the payload to the buffer.
            buf += self.body
            
            # Append a checksum to the end of the body.
            buf += struct.pack("=L", binascii.crc32(self.body) & 0xFFFFFFFF)

        # Return the completed packet buffer.
        return buf

    def unpack_header(self, buffer):
        """
        This function parses a received request header in 'buffer'.
        
        The request header is 28 bytes:
            (4) command
            (4) k
            (8) API key
            (8) body_length
            (4) header checksum 

        :type buffer: bytearray
        :param buffer: The byte array containing the header.
        """
        
        # Unpack the command and query k.
        (self.command, self.k) = struct.unpack_from("=LL", buffer)
            
        # Upnack the 8 character API key, starting at offset 8.
        self.api_key = struct.unpack_from("=s", buffer, 8)[0:8]
        
        # Unpack the body length and checksum.
        (self.body_length, self.checksum) = struct.unpack_from("=QL", buffer, 16)
        
        # Validate the header checksum--compare the transmitted checksum to a
        # checksum of the header (minus the last four bytes of the header, 
        # which are the transmitted checksum!)
        if not self.checksum == (binascii.crc32(buffer[0:-4]) & 0xFFFFFFFF):
            raise IOError("Request header does not match checksum!") 
                
class Response:
    """
    Class representing a response received from the appliance.
    
    The response header is 36 bytes:
        (4) command
        (4) status
        (4) k
        (4) elapsed
        (8) body_length
        (4) header checksum
    
    The body is received over the socket separately from the header.
    First we receive 36 bytes to receive the header, then this tells us
    how many bytes to receive for the body.
    """
    
    def __init__(self, command=0, status=Status.SUCCESS, count=0, elapsed=0, body_length=0,
                 body=None):
        self.command = command
        self.status = status
        self.count = count
        self.elapsed = elapsed
        self.body_length = body_length
        self.checksum = 0
        self.body = body
        self.body_checksum = 0
        
        # Store the header size to be referenced elsewhere in the code.
        self.header_size = 28
    
    def pack(self):
        """
        Pack the response into a byte array (Python string).
        """
        
        # Pack the message header.        
        #   'L' is unsigned long (32-bit, 4 bytes)
        #   'Q' is unsigned long long (64-bit, 8 bytes)
        #   'f' is float (32-bit, 4 bytes)
        buf = struct.pack("=LLLfQ", self.command, self.status, self.count, self.elapsed, self.body_length)
        
        #print('Header checksum: %d' % (binascii.crc32(buf) & 0xFFFFFFFF))        
        
        # Add a checksum of the header fields.       
        buf += struct.pack("=L", binascii.crc32(buf) & 0xFFFFFFFF)

        # Append the packet payload.
        if self.body_length > 0 and self.body is not None:
            buf += self.body
        
            # Add a checksum of the body just after the body.
            buf += struct.pack("=L", binascii.crc32(self.body) & 0xFFFFFFFF)
        
        # Return the constructed packet.
        return buf
        
    def unpack_header(self, buffer):
        """
        Parse the header of a response packet.
        
        :type buffer: bytearray
        :param buffer: The byte array containing the header.
        """
        # Parse the header fields.
        (self.command, self.status, self.count, self.elapsed, self.body_length, self.checksum) = \
            struct.unpack_from("=LLLfQL", buffer, 0)
        
        # Validate the header checksum--compare the transmitted checksum to a
        # checksum of the header (minus the last four bytes of the header, 
        # which are the transmitted checksum!)
        if not self.checksum == (binascii.crc32(buffer[0:-4]) & 0xFFFFFFFF):
            raise IOError("Response header does not match checksum!") 

    def pack_results(self, distances, indeces):
        """
        Create and store a byte string representation of the results.
        
        Sets the 'body' and 'body_length' parameters of this object.
        """
       
        # Store the dimensions of the results matrices first. 
        self.body = struct.pack("=LL", distances.shape[0], distances.shape[1])
       
        # Use built in numpy methods to dump the results to bytes.
        self.body += distances.tobytes() + indeces.tobytes()
                           
        # Store the length of the payload in bytes.
        self.body_length = len(self.body)
            
    def unpack_results(self):
        """
        Restore the results list from the byte string.
        
        :rtype: numpy.ndarray
        :return: (distances, indeces)
        """
        
        # Unpack the matrix shape from the beginning of the body.
        shape = struct.unpack_from("=LL", self.body)
        
        # Calculate the size of each result matrix in bytes.
        matrix_size = shape[0] * shape[1] * 4
        
        start = 8
        end = 8 + matrix_size
        
        # The first half of the body is the matrix of distances.
        # Numpy implements a function for converting from bytes to ndarray.
        distances = np.frombuffer(self.body[start:end], dtype='float32')
        indeces = np.frombuffer(self.body[end:], dtype='int32')
        
        # Reformat the result matrices to their original shape.
        distances = distances.reshape(shape)
        indeces = indeces.reshape(shape)
        
        return distances, indeces
        
