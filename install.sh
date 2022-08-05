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
ipython kernel install --name "weaveio" --user
ipython="$(which ipython)"
pycmd="print('from weaveio import *; data = Data()'); from weaveio import *; data = Data()"
cmd="$ipython --matplotlib -c \\\"$pycmd\\\" -i"
tcshrc_cmd="$ipython --matplotlib -c \"$pycmd\" -i"
echo "alias weaveio=\"$cmd\"" >> "$HOME/.bashrc"
echo "alias weaveio $tcshrc_cmd" >> "$HOME/.tcshrc"
echo "use the cmd \`weaveio\` to enter the weaveio ipython shell"
echo "source your ~/.bashrc or ~/.tcshrc file now to complete installation"
