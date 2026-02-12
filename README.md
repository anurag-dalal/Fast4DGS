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

## TODO
* Calibration with mrcal name and setup apt cams and which port is which
* figure out how to give pose to the arm the ros and the api

## saving calibratin images
sudo /home/anurag/miniconda3/envs/vision/bin/python image_saver.py

## Calibration [mrcal](https://mrcal.secretsauce.net/install.html)
* It more or less assumes a regular grid of N-by-N corners (i.e. N+1-by-N+1 squares)
* It requires all the corners to be observed in order to report the detections from an image. Incomplete chessboard observations aren't supported
```
sudo apt install mrgingham
sudo apt update
sudo apt install mrcal
sudo apt install libmrcal-dev
sudo apt install vnlog libvnlog-dev libvnlog-perl python3-vnlog
sudo apt install feedgnuplot
```

To use with python Building from source is required

Checkboard square in mm: 16.8
num sqaures: 14 x 14

* Detecting corners
    * for port in ports[5000, 5001, .... 8005]:
    * dataset_path: '/home/anurag/Codes/Fast4DGS/dataset/calibration_data/$port/*.png'
    * mrgingham --jobs 4 --gridn 14 dataset_path > '/home/anurag/Codes/Fast4DGS/dataset/calibration/$port/corners.vnl
* Visualization
    * $ < corners.vnl       \
  vnl-filter -p x,y | \
  feedgnuplot --domain --square --set 'xrange [0:6000] noextend' --set 'yrange [3376:0] noextend'
* mrcal-calibrate-cameras \
    --corners-cache corners.vnl \
    --lensmodel LENSMODEL_SPLINED_STEREOGRAPHIC \
    --object-width-n 14 \
    --object-height-n 14 \
    --object-spacing 0.013 \
    --focal 750 \
    --imagersize 1330 1767 \
    --outdir . \
    "*.jpeg"

* mrcal-calibrate-cameras                                                         \
  --corners-cache corners.vnl                                                   \
  --lensmodel LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=18_Ny=30_fov_x_deg=150 \
  --focal 800                                                                  \
  --object-spacing 0.013                                                       \
  --object-width-n 14                                                           \
  '*.jpeg'


# ssh
for ips, pass in [(user1, ip1, pass1), (user2, ip2, pass2), ....]
    ssh user1:ip1
    pass1

    then in ssh terminal
    echo "pass1" | sudo -S shutdown nowsudo pkill -9 gst-launch-1.0
    echo "pass1" | sudo -S shutdown now

## enable wake on lan
* sudo ethtool -s eth0 wol g
* cat /sys/class/net/eth0/address

## remote shutdown
* sudo apt install sshpass
* chmod +x remote_shutdown.sh
* ./remote_shutdown.sh


## wake up all
sudo apt install wakeonlan
wakeonlan -i 10.0.0.255 48:b0:2d:ec:69:32
wakeonlan -i 10.0.0.255 3c:6d:66:02:42:22
wakeonlan -i 10.0.0.255 48:b0:2d:ec:6b:62
wakeonlan -i 10.0.0.255 48:b0:2d:ec:78:46
wakeonlan -i <broadcast-address> <mac-address>