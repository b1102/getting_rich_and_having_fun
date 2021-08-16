conda deactivate
conda remove --name tensor-flow --all

conda env create -f tensor-flow.yml
conda activate tensor-flow

conda remove --name article --all
