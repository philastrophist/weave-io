#!/bin/bash

if which conda; then
        echo "conda already installed"
else
        echo "installing anaconda"
        wget -nc -P /tmp https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash /tmp/Miniconda3-latest-Linux-x86_64.sh
        echo "creating "
        source $HOME/.bashrc
fi
if [[ -z "$(conda env list | grep weaveio)" ]]; then
        echo "creating new environment and installing weaveio"
else
        echo "weaveio environment already installed."
        while true; do
            read -p "Do you wish to delete this weaveio environment and start again? (this is just for you) [y/n] " yn
            case $yn in
                [Yy]* ) conda remove --name weaveio --all --yes; break;;
                [Nn]* ) exit;;
                * ) echo "Please answer yes or no.";;
            esac
        done

fi
conda create --yes --name weaveio python=3.7 numpy scipy matplotlib ipython jupyter
conda activate weaveio 
pip install graphviz weaveio --no-cache-dir
python -c "from weaveio import *" # test installation

echo "use the cmd \`weaveio\` to enter the weaveio ipython shell"
echo "remove any aliases in your bashrc file to allow the new weaveio command to be used"
echo "source your ~/.bashrc or ~/.tcshrc file now to complete installation"
