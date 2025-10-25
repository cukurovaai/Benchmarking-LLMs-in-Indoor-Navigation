#!/bin/bash
pip install -r requirements.txt

conda install habitat-sim=0.2.2 -c conda-forge -c aihabitat -y

cd ~
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/

# switch to a compatible version of hloc, see the commit: https://github.com/cvg/Hierarchical-Localization/commit/936040e8d67244cc6c8c9d1667701f3ce87bf075
git checkout 936040e8d67244cc6c8c9d1667701f3ce87bf075 

python -m pip install -e .
