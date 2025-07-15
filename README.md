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

## Acknowledgment
This repository is primarily built upon [Visual Language Maps (VLMaps)](https://vlmaps.github.io/). We sincerely thank [Huang et al.](https://arxiv.org/pdf/2210.05714) for releasing their work as open-source.
