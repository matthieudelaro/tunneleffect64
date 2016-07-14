# docker build -t matthieudelaro/tunneleffect64 .

FROM matthieudelaro/caffe-iuf-python

# COPY ../hyperestimate/dataset /dataset
COPY ./dataset /dataset

# COPY . /opt/nut/tunneleffect/
COPY ./src /opt/nut/tunneleffect/src
COPY ./VisualizeReconstructionOfImage.py /opt/nut/tunneleffect/VisualizeReconstructionOfImage.py
COPY ./proto /opt/nut/tunneleffect/proto
RUN mkdir -p /opt/nut/tunneleffect/logs /opt/nut/tunneleffect/snapshots

COPY ./tunnel.sh /opt/nut/.deploy/tunnel.sh

ENTRYPOINT /opt/nut/.deploy/tunnel.sh && /bin/bash
WORKDIR /opt/nut/tunneleffect/

ENV NUT_VOLUME_MAIN_PATH /opt/nut/tunneleffect/
ENV NUT_VOLUME_DATASET_PATH /opt/nut/tunneleffect/dataset

# Then run with nvidia-docker:
# nvidia-docker run -it --rm --workdir=/nut/tunneleffect/ --volume=/home/matthieudelaro/Documents/projects/tunnelEffect3.5.pretrainCifar10:/nut/tunneleffect --volume=/home/matthieudelaro/Documents/projects/hyperestimate/dataset:/dataset --env="NUT_VOLUME_MAIN_PATH=/nut/tunneleffect/" --env="NUT_VOLUME_DATASET_PATH=/nut/tunneleffect/dataset"  matthieudelaro/tunneleffect64

# Or rely on nvidia-docker's driver, but specify all params by yourself:
# docker run -it --rm --workdir=/nut/tunneleffect/ --volume=/home/matthieudelaro/Documents/projects/tunnelEffect3.5.pretrainCifar10:/nut/tunneleffect --volume=/home/matthieudelaro/Documents/projects/hyperestimate/dataset:/dataset --volume-driver=nvidia-docker --volume=/var/lib/docker/volumes/nvidia_driver_352.63/_data:/usr/local/nvidia:ro --env="NUT_VOLUME_MAIN_PATH=/nut/tunneleffect/" --env="NUT_VOLUME_DATASET_PATH=/nut/tunneleffect/dataset" --device="/dev/nvidia0:/dev/nvidia0:mrw" --device="/dev/nvidiactl:/dev/nvidiactl:mrw" --device="/dev/nvidia-uvm:/dev/nvidia-uvm:mrw" matthieudelaro/tunneleffect64

# Generate this command:
# nut --dockercli --macro=tunnel --logs => copy the full command from the logs
# change the docker image name
# in case of this problem, "docker: Error response from daemon: create nvidia_driver_352.63: conflict: volume name must be unique.""
# replace the volume name with its path (Mountpoint given by docker volume inspect nvidia_driver_352.63)
