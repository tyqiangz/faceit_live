docker run --runtime=nvidia --init -it --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v $PWD:/code/faceit_live  -w=/code/faceit_live -v ~/.Xauthority:/root/.Xauthority faceit:latest




# python faceit_live.py live ale_to_oliver
