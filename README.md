## Benchmarking Large Language Model Reasoning in Indoor Robot Navigation
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://cukurovaai.github.io/Bench-LLM-Nav/)

[Emirhan Balcı](https://emirhanbalci.me/), [Mehmet Sarıgül](http://mehmetsarigul.com/), [Barış Ata](https://barisata.me/)

This study evaluates the performance of state-of-the-art text-based generative large language models in indoor robot navigation planning, focusing on object, spatial, and common-sense reasoning-centric instructions. Three scenes from the Matterport3D dataset were selected, along with corresponding instruction sequences and routes. Object-labeled semantic maps were generated using the RGB-D images and camera poses of the scenes. The instructions were provided to the models, and the generated robot codes were executed on a mobile robot within the selected scenes. The routes followed by the robot, which detected objects through the semantic map, were recorded. The findings indicate that while the models successfully executed object and spatial-based instructions, some models struggled with those requiring common-sense reasoning. This study aims to contribute to robotics research by providing insights into the navigation planning capabilities of language models.

![](media/project_scheme_transparent.png)
