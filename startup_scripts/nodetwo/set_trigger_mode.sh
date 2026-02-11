#!/bin/bash

sleep 60

#Automatically detect start and end /dev/videoN

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

# Configure cameras for external trigger mode at 20fps
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

# Wait for all background processes to finish
wait