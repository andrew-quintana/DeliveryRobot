#!/bin/bash

# workaround for ssh issues when using command line 

# move to home dir
cd ~

# remove project dir
rm -rf DeliveryRobot

# get the branch's updates
git clone --single-branch --branch hardware_integration https://github.com/andrew-quintana/DeliveryRobot.git