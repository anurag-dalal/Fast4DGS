torch                       2.9.0+cu126
gsplat                      1.4.0

## command to launch
gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! autovideosink sync=false

## startup scripts
* ntp server  ✔
* pwm (only for 1) ✔
* set all cameras to ext rigger ✔
* gstreamer streaming code
* set all cameras to same configs like exposure and all