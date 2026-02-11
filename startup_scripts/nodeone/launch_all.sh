#!/bin/bash

# Variables
HOST_IP="10.0.0.24"
BITRATE=10000000         # Default bitrate
START_PORT=5000         # Default starting port
START_DEVICE=0          # Start with /dev/video0
END_DEVICE=5            # End with /dev/video5

# Override defaults from command-line arguments
if [ ! -z "$1" ]; then
  HOST_IP=$1
fi
if [ ! -z "$2" ]; then
  BITRATE=$2
fi
if [ ! -z "$3" ]; then
  START_PORT=$3
fi

# Loop through video devices
for i in $(seq $START_DEVICE $END_DEVICE); do
  DEVICE="/dev/video$i"
  PORT=$((START_PORT + i))

  echo "Launching pipeline for $DEVICE to $HOST_IP:$PORT with bitrate $BITRATE"

  gst-launch-1.0 nvv4l2camerasrc device=$DEVICE ! \
    "video/x-raw(memory:NVMM), format=(string)UYVY, width=(int)1920, height=(int)1080" ! \
    queue ! nvvidconv ! "video/x-raw(memory:NVMM), format=(string)I420" ! \
    nvv4l2h264enc bitrate=$BITRATE iframeinterval=10 insert-sps-pps=true profile=1 ! \
    rtph264pay config-interval=1 pt=96 ! \
    udpsink clients=$HOST_IP:$PORT sync=true &

done

# Wait for all background pipelines to finish
wait


# Working, but not synced:   gst-launch-1.0 nvv4l2camerasrc device=$DEVICE ! \
#    "video/x-raw(memory:NVMM), format=(string)UYVY, width=(int)1920, height=(int)1080, framerate=(fraction)20/1" ! \
#    queue ! nvvidconv ! "video/x-raw(memory:NVMM), format=(string)I420" ! \
#    nvv4l2h264enc bitrate=$BITRATE iframeinterval=10 insert-sps-pps=true profile=1 ! \
#    rtph264pay config-interval=1 pt=96 ! \
#    udpsink clients=$HOST_IP:$PORT &