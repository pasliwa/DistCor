#!/usr/bin/env bash

python ../create_data_background.py 500 0 5 ../network_specification/net2-50-quadratic.txt ~/DistanceCor/network_backgrounds/net2/ 0 6
python ../create_data_background.py 500 0 5 ../network_specification/net3-50-quadratic.txt ~/DistanceCor/network_backgrounds/net3/ 0 10
python ../create_data_background.py 500 0 5 ../network_specification/net2-10-quadratic.txt ~/DistanceCor/network_backgrounds/net2/ 0 6
python ../create_data_background.py 500 0 5 ../network_specification/net2-30-quadratic.txt ~/DistanceCor/network_backgrounds/net2/ 0 6


