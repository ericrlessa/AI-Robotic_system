#!/bin/bash

gst-launch-1.0 libcamerasrc \
  ! video/x-raw,width=640,height=480,framerate=30/1,format=I420 \
  ! videoconvert \
  ! x264enc speed-preset=ultrafast tune=zerolatency byte-stream=true key-int-max=75 \
  ! video/x-h264,level='(string)4' \
  ! h264parse \
  ! video/x-h264,stream-format=avc,alignment=au,width=640,height=480,framerate=30/1 \
  ! kvssink stream-name=pi-stream-object-detection aws-region=us-east-1 iot-certificate=iot-certificate,endpoint=c3caokm6ip3jfc.credentials.iot.us-east-1.amazonaws.com,cert-path=/home/user/AI-Robotic_system_Feb2025/cloud/aws/certs/pi-certificate.pem.crt,key-path=/home/user/AI-Robotic_system_Feb2025/cloud/aws/certs/pi-private.pem.key,ca-path=/home/user/AI-Robotic_system_Feb2025/cloud/aws/certs/AmazonRootCA1.pem,role-aliases=CameraIOTRoleAlias