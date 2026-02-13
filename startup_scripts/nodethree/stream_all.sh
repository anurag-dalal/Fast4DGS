#!/bin/bash
sleep 20
### ------------ Automatically detect start and end /dev/videoN
shopt -s nullglob
video_nodes=(/dev/video*)
shopt -u nullglob

if [ ${#video_nodes[@]} -eq 0 ]; then
  echo "No /dev/video* devices found" >&2
  exit 1
fi

idxs=()
for dev in "${video_nodes[@]}"; do
  n=$(basename "$dev" | sed -E 's/^video//')
  if [[ $n =~ ^[0-9]+$ ]]; then
    idxs+=("$n")
  fi
done

IFS=$'\n' sorted=($(printf '%s\n' "${idxs[@]}" | sort -n))
unset IFS
START_DEVICE=${sorted[0]}
END_DEVICE=${sorted[$((${#sorted[@]}-1))]}

### ------------ Configure cameras for external trigger mode at 20fps
echo "Configuring cameras for external trigger mode..."
for i in $(seq $START_DEVICE $END_DEVICE); do
  DEVICE="/dev/video$i"
  echo "Setting up $DEVICE for external trigger"

  # Set trigger mode to external (1)
  v4l2-ctl -d $DEVICE --set-ctrl=trigger=1

  # Set exposure time for 20fps (50ms max frame time, use shorter exposure)
  v4l2-ctl -d $DEVICE --set-ctrl=exposure_time_absolute=100

  # Set manual exposure mode
  v4l2-ctl -d $DEVICE --set-ctrl=exposure_auto=1

  echo "Camera $DEVICE configured for external trigger"
done

#### ------------ START GSTREAMER UDP STREAM TO HOST_IP 
#### ------------ CONFIGURE THE START PORT
# Variables
HOST_IP="10.0.0.24"
BITRATE=20000000         # Default bitrate
START_PORT=7000         # Default starting port
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
  queue max-size-buffers=3 ! nvvidconv ! "video/x-raw(memory:NVMM), format=(string)I420" ! \
  nvv4l2h265enc bitrate=$BITRATE  preset-level=4 \
    idrinterval=15 insert-sps-pps=true insert-vui=true \
    control-rate=1 ratecontrol-enable=true ! \
  h265parse config-interval=1 ! \
  rtph265pay config-interval=1 pt=96 mtu=1400 ! \
  udpsink clients=$HOST_IP:$PORT sync=false async=false &



done

# Wait for all background pipelines to finish
wait

# Wait for all background processes to finish
wait