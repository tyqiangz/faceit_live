
# New Version -> FaceIt Live3

Checkout my other repository [Faceit Live3](https://github.com/alew3/faceit_live3) for a more up to date version, that uses a new technique that only requires a single image from the target image and no training is required. 

# FaceIt Live

FaceIT Live will swap your face in realtime to someone else's. For laughs you can enter a conference with this modified video stream.

![Joing a video conference call as John Oliver or another celebrity is great fun](deep_live.gif)

This code is based on the library [deepfakes/faceswap](https://github.com/deepfakes/faceswap) and the work done by Gaurav Oberoi on the [FaceIt](https://github.com/goberoi/faceit) library that makes it easy to extract frames for training directly from YouTube.

# Setup

## Requirements
This has been tested on **Ubuntu 16.04 with a Titan X (Pascal) GPU**.
You will need the following to make it work:

    Linux host OS
    NVidia fast GPU (GTX 1080, GTX 1080i, Titan, etc ...)
    Fast Desktop CPU (Quad Core or more)
    NVidia CUDA 9 and cuDNN 7 libraries installed
    Docker installed
    Webcam working at /dev/video0

## Setup Host System
To use the fake webcam feature to enter conferences with our stream we need to insert the **v4l2loopback** kernel module. Follow the install instructions at  (https://github.com/umlaeute/v4l2loopback), then let's setup our fake webcam:

```
$ git clone https://github.com/umlaeute/v4l2loopback.git
$ make && sudo make install
$ sudo depmod -a
$ sudo modprobe v4l2loopback video_nr=1
$ v4l2-ctl -d /dev/video1 -c timeout=3000
```

Via PPA “Oficial”
sudo add-apt-repository ppa:webcamstudio/webcamstudio-dailybuilds
sudo apt-get update
sudo apt-get instal webcamstudio

This will create a new stream at */dev/video1*

## Clone this repository
Don't forget to use the *--recurse-submodules* parameter to checkout all dependencies.

    $ git clone --recurse-submodules https://github.com/alew3/faceit_live.git /local_path/

## Setup Docker
To make it easy to install all depencies a Dockerfile has been provided. After [installing Docker](https://docs.docker.com/install/).  Go to the project directory and:
    
    $ cd /local_path/faceit_live
    $ xhost local:root # this is necessary for your docker to access the host interface
    $ docker-compose build
    $ chmod +x ./run_docker.sh

To run the docker use the provided shell script that runs the Docker and makes the webcam and XTerminal available in the container.

    $ ./run_docker.sh

# Usage


Then create the directory `./data/persons` and put one image containing the face of person A and another of person B. Use the same name that you did when setting up the model. This file is used to filter their face from any others in the videos you provide. E.g.:
```
./data/persons/me.jpg
./data/persons/oliver.jpg
```

# Capture a video sample of you from your webcam

You can use **Cheese** or another program to capture a video from your webcam and use that for training. This is the better than using a mobile phone you will be in your real world environment that will be later used for conversion.

    $ sudo apt-get install cheese
    $ cheese

This code is based on the library [deepfakes/faceswap](https://github.com/deepfakes/faceswap) and the work done by Gaurav Oberoi on the [FaceIt](https://github.com/goberoi/faceit) library that makes it easy to extract frames for training directly from YouTube.

You will need at least 512 images for training, so do a few videos from yourself moving your head around and making different expressions. Afterwards put them in the folder ./data/videos/:

    e.g.
    ./data/videos/myvideo1.webm
    ./data/videos/myvideo2.webm


For the training videos from the second person you may either use videos from Youtube or put them directly into the same **./data/videos** folder. 

Setup your model and training data in code on the file **faceit_live.py**, e.g.:
```python
# Create the model with params: model name, person A name, person B name.
faceit = FaceIt('me_to_oliver', 'me', 'oliver')

# Add your videos from  YouTube url or filename of the video (in folder /data/videos).
faceit.add_video('me', 'myvideo1.webm')
faceit.add_video('me', 'myvideo2.webm')
faceit.add_video('me', 'me_from_youtube.mp4', 'youtube url here')

# Do the same for person B.
faceit.add_video('oliver', 'oliver_trumpcard.mp4', 'https://www.youtube.com/watch?v=JlxQ3IUWT0I')
faceit.add_video('oliver', 'oliver_taxreform.mp4', 'https://www.youtube.com/watch?v=g23w7WPSaU8')
faceit.add_video('oliver', 'oliver_zazu.mp4', 'https://www.youtube.com/watch?v=Y0IUPwXSQqg')
```

Now let's startup our docker container and run our code:
```
$ ./run_docker.sh
```


Then, preprocess the data. This downloads the videos, breaks them into frames, and extracts the relevant faces. After running the script, go to ./data/processed/ to make sure it didn't extract a different persons images into your training data. If it did, just delete them.
```
python faceit_live.py preprocess me_to_oliver
```

Then train the model, e.g.:
```
python faceit_live.py train me_to_oliver
```

To see how well it is working use:
```
python faceit_live.py live me_to_oliver
```

# Let's join a video conference!
To create a fake webcam stream use the following and select "Dummy" as your webcam source with Firefox and a WebRTC videoconferencing site such as http://www.appear.in/ , make sure to use a select a separate audio source or people won't hear you.
```
python faceit_live.py webcam me_to_oliver
```


If you prefer, you can still convert an existing video that is stored on disk, e.g.:
```
python faceit_live.py convert ale_to_oliver myvideo1.webm --start 10 --duration 20 --side-by-side
```


## TODO
- Check if there is a fix for Skype and Chrome not recognizing the webcam
