# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0.

'''
Example to demonstrate usage the AWS Kinesis Video Streams (KVS) Consumer Library for Python.
 '''
 
__version__ = "0.0.1"
__status__ = "Development"
__copyright__ = "Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved."
__author__ = "Dean Colcott <https://www.linkedin.com/in/deancolcott/>"

import os
import sys
import time
import boto3
import requests
import logging
from src.kinesis_video_streams_parser import KvsConsumerLibrary
from src.kinesis_video_fragment_processor import KvsFragementProcessor
import subprocess
from ultralytics import YOLO
import cv2


# Config the logger.
log = logging.getLogger(__name__)
logging.basicConfig(format="[%(name)s.%(funcName)s():%(lineno)d] - [%(levelname)s] - %(message)s", 
                    stream=sys.stdout, 
                    level=logging.INFO)

# Update the desired region and KVS stream name.
REGION = os.environ.get('AWS_REGION', 'us-east-1')
KVS_STREAM01_NAME = os.environ.get('STREAM', 'pi-stream-object-detection')

class KvsPythonConsumerExample:
    '''
    Example class to demonstrate usage the AWS Kinesis Video Streams KVS) Consumer Library for Python.
    '''

    def __init__(self):
        '''
        Initialize the KVS clients as needed. The KVS Comsumer Library intentionally does not abstract 
        the KVS clients or the various media API calls. These have individual authentication configuration and 
        a variety of other user defined settings so we keep them here in the users application logic for configurability.

        The KvsConsumerLibrary sits above these and parses responses from GetMedia and GetMediaForFragmentList 
        into MKV fragments and provides convenience functions to further process, save and extract individual frames.  
        '''

        # Create shared instance of KvsFragementProcessor
        self.kvs_fragment_processor = KvsFragementProcessor()

        # Variable to maintaun state of last good fragememt mostly for error and exception handling.
        self.last_good_fragment_tags = None

        # Init the KVS Service Client and get the accounts KVS service endpoint
        log.info('Initializing Amazon Kinesis Video client....')
        # Attach session specific configuration (such as the authentication pattern)
        self.session = boto3.Session(region_name=REGION)
        self.kvs_client = self.session.client("kinesisvideo")

    ####################################################
    # Main process loop
    def service_loop(self):
        
        ####################################################
        # Start an instance of the KvsConsumerLibrary reading in a Kinesis Video Stream

        # Get the KVS Endpoint for the GetMedia Call for this stream
        log.info(f'Getting KVS GetMedia Endpoint for stream: {KVS_STREAM01_NAME} ........') 
        get_media_endpoint = self._get_data_endpoint(KVS_STREAM01_NAME, 'GET_MEDIA')
        
        # Get the KVS Media client for the GetMedia API call
        log.info(f'Initializing KVS Media client for stream: {KVS_STREAM01_NAME}........') 
        kvs_media_client = self.session.client('kinesis-video-media', endpoint_url=get_media_endpoint)

        # Make a KVS GetMedia API call with the desired KVS stream and StartSelector type and time bounding.
        log.info(f'Requesting KVS GetMedia Response for stream: {KVS_STREAM01_NAME}........') 
        get_media_response = kvs_media_client.get_media(
            StreamName=KVS_STREAM01_NAME,
            StartSelector={
                'StartSelectorType': 'NOW'
            }
        )
        
        # Initialize an instance of the KvsConsumerLibrary, provide the GetMedia response and the required call-backs
        log.info(f'Starting KvsConsumerLibrary for stream: {KVS_STREAM01_NAME}........') 
        my_stream01_consumer = KvsConsumerLibrary(KVS_STREAM01_NAME, 
                                                    get_media_response, 
                                                    self.on_fragment_arrived, 
                                                    self.on_stream_read_complete, 
                                                    self.on_stream_read_exception
                                                    )

        print("starting consumer")
        my_stream01_consumer.start()
        print("joining consumer...")
        my_stream01_consumer.join()
            

    ####################################################
    # KVS Consumer Library call-backs

    def on_fragment_arrived(self, stream_name, fragment_bytes, fragment_dom, fragment_receive_duration):
        try:
            self.last_good_fragment_tags = self.kvs_fragment_processor.get_fragment_tags(fragment_dom)

            one_in_frames_ratio = 5
            self.kvs_fragment_processor.process_frame_to_robot(fragment_bytes, one_in_frames_ratio)

        except Exception as err:
            log.error(f'on_fragment_arrived Error: {err}')
    
    
    
    def on_stream_read_complete(self, stream_name):
        '''
        This callback is triggered by the KvsConsumerLibrary when a stream has no more fragments available.
        This represents a graceful exit of the KvsConsumerLibrary thread.

        A stream will reach the end of the available fragments if the StreamSelector applied some 
        time or fragment bounding on the media request or if requesting a live steam and the producer 
        stopped sending more fragments. 

        Here you can choose to either restart reading the stream at a new time or just clean up any
        resources that were expecting to process any further fragments. 
        
        ### Parameters:

            **stream_name**: str
                Name of the stream as set when the KvsConsumerLibrary thread triggering this callback was initiated.
                Use this to identify a fragment when multiple streams are read from different instances of KvsConsumerLibrary to this callback.
        '''

        # Do something here to tell the application that reading from the stream ended gracefully.
        print(f'Read Media on stream: {stream_name} Completed successfully - Last Fragment Tags: {self.last_good_fragment_tags}')

    def on_stream_read_exception(self, stream_name, error):
        '''
        This callback is triggered by an exception in the KvsConsumerLibrary reading a stream. 
        
        For example, to process use the last good fragment number from self.last_good_fragment_tags to
        restart the stream from that point in time with the example stream selector provided below. 
        
        Alternatively, just handle the failed stream as per your application logic requirements.

        ### Parameters:

            **stream_name**: str
                Name of the stream as set when the KvsConsumerLibrary thread triggering this callback was initiated.
                Use this to identify a fragment when multiple streams are read from different instances of KvsConsumerLibrary to this callback.

            **error**: err / exception
                The Exception obje tvthat was thrown to trigger this callback.

        '''

        # Can choose to restart the KvsConsumerLibrary thread at the last received fragment with below example StartSelector
        #StartSelector={
        #    'StartSelectorType': 'FRAGMENT_NUMBER',
        #    'AfterFragmentNumber': self.last_good_fragment_tags['AWS_KINESISVIDEO_CONTINUATION_TOKEN'],
        #}

        # Here we just log the error 
        print(f'####### ERROR: Exception on read stream: {stream_name}\n####### Fragment Tags:\n{self.last_good_fragment_tags}\nError Message:{error}')

    ####################################################
    # KVS Helpers
    def _get_data_endpoint(self, stream_name, api_name):
        '''
        Convenience method to get the KVS client endpoint for specific API calls. 
        '''
        response = self.kvs_client.get_data_endpoint(
            StreamName=stream_name,
            APIName=api_name
        )
        return response['DataEndpoint']

if __name__ == "__main__":
    '''
    Main method for example KvsConsumerLibrary
    '''
    
    kvsConsumerExample = KvsPythonConsumerExample()
    
    while True:
        kvsConsumerExample.service_loop()
