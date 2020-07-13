#!/usr/bin/env bash

python ../create_data_background.py 500 0 5 ../network_specification/net1-30-quadratic.txt ~/DistanceCor/network_backgrounds/net1/ 0 10
python ../create_data_background.py 500 0 5 ../network_specification/net1-50-quadratic.txt ~/DistanceCor/network_backgrounds/net1/ 0 10
python ../create_data_background.py 500 0 5 ../network_specification/net1-10-quadratic.txt ~/DistanceCor/network_backgrounds/net1/ 0 10

