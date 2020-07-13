#!/usr/bin/env bash

python ../create_data_background.py 500 0 5 ../network_specification/net3-50-sin.txt ~/DistanceCor/network_backgrounds/net3/ 0 10
python ../create_data_background.py 500 0 5 ../network_specification/net3-10-quadratic.txt ~/DistanceCor/network_backgrounds/net3/ 0 10
python ../create_data_background.py 500 0 5 ../network_specification/net3-10-sin.txt ~/DistanceCor/network_backgrounds/net3/ 0 10
python ../create_data_background.py 500 0 5 ../network_specification/net3-30-sin.txt ~/DistanceCor/network_backgrounds/net3/ 0 10

