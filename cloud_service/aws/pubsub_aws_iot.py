from awscrt import mqtt, http
from awsiot import mqtt_connection_builder
import sys
import threading
import time
import json
from dataclasses import dataclass
import os

# This sample uses the Message Broker for AWS IoT to send and receive messages
# through an MQTT connection. On startup, the device connects to the server,
# subscribes to a topic, and begins publishing messages to that topic.
# The device should receive those same messages back from the message broker,
# since it is subscribed to that same topic.

received_count = 0
received_all_event = threading.Event()

# Callback when connection is accidentally lost.
def on_connection_interrupted(connection, error, **kwargs):
    print("Connection interrupted. error: {}".format(error))


# Callback when an interrupted connection is re-established.
def on_connection_resumed(connection, return_code, session_present, **kwargs):
    print("Connection resumed. return_code: {} session_present: {}".format(return_code, session_present))

    if return_code == mqtt.ConnectReturnCode.ACCEPTED and not session_present:
        print("Session did not persist. Resubscribing to existing topics...")
        resubscribe_future, _ = connection.resubscribe_existing_topics()

        # Cannot synchronously wait for resubscribe result because we're on the connection's event-loop thread,
        # evaluate result with a callback instead.
        resubscribe_future.add_done_callback(on_resubscribe_complete)


def on_resubscribe_complete(resubscribe_future):
    resubscribe_results = resubscribe_future.result()
    print("Resubscribe results: {}".format(resubscribe_results))

    for topic, qos in resubscribe_results['topics']:
        if qos is None:
            sys.exit("Server rejected resubscribe to topic: {}".format(topic))

# Callback when the connection successfully connects
def on_connection_success(connection, callback_data):
    assert isinstance(callback_data, mqtt.OnConnectionSuccessData)
    print("Connection Successful with return code: {} session present: {}".format(callback_data.return_code, callback_data.session_present))

# Callback when a connection attempt fails
def on_connection_failure(connection, callback_data):
    assert isinstance(callback_data, mqtt.OnConnectionFailureData)
    print("Connection failed with error code: {}".format(callback_data.error))

# Callback when a connection has been disconnected or shutdown successfully
def on_connection_closed(connection, callback_data):
    print("Connection closed")

from pathlib import Path

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.__init_once__()
        return cls._instance

    def __init_once__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.endPoint = "ai6mf8hjrnusz-ats.iot.us-east-1.amazonaws.com"
        self.cert = os.path.join(current_dir, "certs/pi-certificate.pem.crt")
        self.key = os.path.join(current_dir, "certs/pi-private.pem.key")
        self.ca = os.path.join(current_dir, "certs/AmazonRootCA1.pem")        

def get_connection(client_id_param):
    config = Config()

    mqtt_connection = mqtt_connection_builder.mtls_from_path(
        endpoint=config.endPoint,
        port=8883,
        cert_filepath=config.cert,
        pri_key_filepath=config.key,
        ca_filepath=config.ca,
        on_connection_interrupted=on_connection_interrupted,
        on_connection_resumed=on_connection_resumed,
        client_id=client_id_param,
        clean_session=False,
        keep_alive_secs=30,
        http_proxy_options=None,        
        on_connection_success=on_connection_success,
        on_connection_failure=on_connection_failure,
        on_connection_closed=on_connection_closed)

    print("Connecting to endpoint")
    connect_future = mqtt_connection.connect()

    # Future.result() waits until a result is available
    connect_future.result()
    print("Connected!")
    
    return mqtt_connection

def publish(mqtt_connection, message, message_topic='topic/robot_command'):
    print("Publishing message to topic '{}': {}".format(message_topic, message))
    
    #message_json = json.dumps(message)
    mqtt_connection.publish(
        topic=message_topic,
        payload=message,
        qos=mqtt.QoS.AT_LEAST_ONCE)

# Define the subscribe function to accept a custom callback method
def subscribe(client_id, callback, message_topic='topic/robot_command'):
    print("subscribe message topic '{}'".format(message_topic))

    mqtt_connection = get_connection(client_id)

    subscribe_future, packet_id = mqtt_connection.subscribe(
        topic=message_topic,
        qos=mqtt.QoS.AT_LEAST_ONCE,
        callback=lambda topic, payload, dup, qos, retain, **kwargs: on_message_received(topic, payload, callback)
    )

    subscribe_result = subscribe_future.result()
    print("Subscribed with {}".format(str(subscribe_result['qos'])))

    return mqtt_connection

# Callback when the subscribed topic receives a message
def on_message_received(topic, payload, callback):
    print(f"Received message from topic '{topic}': {payload.decode()}")
    
    # Execute the callback function passed as a parameter
    if callback:
        callback(payload.decode())

