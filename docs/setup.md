# Setup

## Prerequisites
The WEAVE-IO DB can only be hosted on the Hertfordshire UHHPC. 
As such any potential user of WEAVE-IO will need obtain a UHHPC username under the WEAVE-LOFAR group.
Contact Martin Hardcastle to get setup. 

## Installation 

WEAVE-IO requires Python 3 to function, use Python 2 at your own risk.

To install use `pip install weaveio`

## Setup SSH tunnel to UHHPC
Firstly, setup SSH key authentication:

1. Setup the alias in your ssh configuration by copy and pasting the following into `~/.ssh/config` (setting your username):

        Host uhhpc
                HostName uhhpc.herts.ac.uk
                User $user
                IdentityFile ~/.ssh/id_rsa
                LocalForward 11007 127.0.0.1:11007
                LocalForward 11008 127.0.0.1:11008
                LocalForward 7473 127.0.0.1:7473

2. Generate ssh key: `ssh-keygen` and follow the instructions
3. `ssh $user@uhhpc.herts.ac.uk` to start the port forwarding

You need to run `ssh $user@uhhpc.herts.ac.uk` once everytime you start work on 