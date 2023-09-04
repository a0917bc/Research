#!/bin/bash
# cd ~/Research
# python prototype_learning.py --num 1024

# Loop from 0 to 11
for i in {0..11}
do
  echo "Running script for layer $i"
  python prototype_learning.py --layer $i --stop 12 --num 1024
done
