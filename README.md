# Benchmarking Large Language Model Reasoning in Indoor Robot Navigation
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://cukurovaai.github.io/Bench-LLM-Nav/)
[![Static Badge](https://img.shields.io/badge/Project-Poster-blue)](https://emirhanbalci.me/docs/SIU_Poster.pdf)
[![Static Badge](https://img.shields.io/badge/Project-Video-orange)](https://youtu.be/ddHJjeE96u8)

[Emirhan Balcı](https://emirhanbalci.me/), [Mehmet Sarıgül](http://mehmetsarigul.com/), [Barış Ata](https://barisata.me/)

This study evaluates the performance of state-of-the-art text-based generative large language models in indoor robot navigation planning, focusing on object, spatial, and common-sense reasoning-centric instructions. Three scenes from the Matterport3D dataset were selected, along with corresponding instruction sequences and routes. Object-labeled semantic maps were generated using the RGB-D images and camera poses of the scenes. The instructions were provided to the models, and the generated robot codes were executed on a mobile robot within the selected scenes. The routes followed by the robot, which detected objects through the semantic map, were recorded. The findings indicate that while the models successfully executed object and spatial-based instructions, some models struggled with those requiring common-sense reasoning. This study aims to contribute to robotics research by providing insights into the navigation planning capabilities of language models.

![](media/project_scheme_transparent.png)

# Approach

<p align="center">
  <img src="media/scheme_animated.gif" alt="Demo GIF" width="900">
</p>

# Acknowledgment
This repository is primarily built upon [Visual Language Maps (VLMaps)](https://vlmaps.github.io/). We sincerely thank [Huang et al.](https://arxiv.org/pdf/2210.05714) for releasing their work as open-source.

# Installation
> [!IMPORTANT]
> The source code was developed and tested inside a Docker container based on an Ubuntu image to isolate project dependencies. Therefore, the installation steps assume that Docker (version 27.0.3+) and X11 are installed on a Linux machine with an NVIDIA GPU for full compatibility.
> Visual outputs can be displayed via the X11 display protocol.

### Download Matterpot3D Dataset
To create a realistic environment for indoor robot navigation tasks, we leveraged the Matterport3D dataset, comprising high-resolution RGB-D images and 3D reconstructions of real-world indoor spaces. Please check [Dataset Download](https://niessner.github.io/Matterport/), sign the [Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf), and send it to the responsible person to request the Matterport3D mesh for use in the Habitat simulator. The return email will attach a Python script to download the data. Copy and paste the script into a file `~/download_mp.py`. 

### Docker Setup :whale2:
Download the `Dockerfile` in the root of this repository and place it in the same directory as the `~/download_mp.py` script. Alternatively, you can copy and paste its contents into a new file named `Dockerfile`. Ensure that the `~/download_mp.py` script is located in the same directory, as it is required during the build process.

Once both files are in place, build the Docker image with:

```bash
# build the docker image
docker build -t "Bench_LLM_Nav" .
```

To run the container with GUI and GPU support, use the following command:

```bash
docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/<your_username>/.Xauthority:/root/.Xauthority --net=host --ipc=host --runtime=nvidia --gpus all Bench_LLM_Nav
```

