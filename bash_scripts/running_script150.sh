#!/usr/bin/env bash

python ../create_data_background.py 500 0 5 ../network_specification/net1-150-linear.txt ~/DistanceCor/network_backgrounds/net1/ 0 10 &
python ../create_data_background.py 500 0 5 ../network_specification/net1-150-quadratic.txt ~/DistanceCor/network_backgrounds/net1/ 0 10 &
python ../create_data_background.py 500 0 5 ../network_specification/net1-150-sin.txt ~/DistanceCor/network_backgrounds/net1/ 0 10 &

python ../create_data_background.py 500 0 5 ../network_specification/net2-150-linear.txt ~/DistanceCor/network_backgrounds/net2/ 0 6 &
python ../create_data_background.py 500 0 5 ../network_specification/net2-150-quadratic.txt ~/DistanceCor/network_backgrounds/net2/ 0 6 &
python ../create_data_background.py 500 0 5 ../network_specification/net2-150-sin.txt ~/DistanceCor/network_backgrounds/net2/ 0 6 &

python ../create_data_background.py 500 0 5 ../network_specification/net3-150-linear.txt ~/DistanceCor/network_backgrounds/net3/ 0 10 &
python ../create_data_background.py 500 0 5 ../network_specification/net3-150-quadratic.txt ~/DistanceCor/network_backgrounds/net3/ 0 10 &
python ../create_data_background.py 500 0 5 ../network_specification/net3-150-sin.txt ~/DistanceCor/network_backgrounds/net3/ 0 10 &

wait 

python ../create_data_background.py 500 0 7 ../network_specification/net4-150-linear.txt ~/DistanceCor/network_backgrounds/net4/ 0 9 &
python ../create_data_background.py 500 0 7 ../network_specification/net4-150-quadratic.txt ~/DistanceCor/network_backgrounds/net4/ 0 9 &
python ../create_data_background.py 500 0 7 ../network_specification/net4-150-sin.txt ~/DistanceCor/network_backgrounds/net4/ 0 9 &

python ../create_data_background.py 500 0 7 ../network_specification/net4-150-linear.txt ~/DistanceCor/network_backgrounds/net4/ 9 19 &
python ../create_data_background.py 500 0 7 ../network_specification/net4-150-quadratic.txt ~/DistanceCor/network_backgrounds/net4/ 9 19 &
python ../create_data_background.py 500 0 7 ../network_specification/net4-150-sin.txt ~/DistanceCor/network_backgrounds/net4/ 9 19 &

python ../create_data_background.py 500 0 7 ../network_specification/net4-150-linear.txt ~/DistanceCor/network_backgrounds/net4/ 19 28 &
python ../create_data_background.py 500 0 7 ../network_specification/net4-150-quadratic.txt ~/DistanceCor/network_backgrounds/net4/ 19 28 &
python ../create_data_background.py 500 0 7 ../network_specification/net4-150-sin.txt ~/DistanceCor/network_backgrounds/net4/ 19 28 &

