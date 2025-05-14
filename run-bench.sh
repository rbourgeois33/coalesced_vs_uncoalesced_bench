#!/bin/bash

rm *.txt

# Step 1: Run ./main --scale_test and capture the integer output
#./main --scale_test
X=28

echo "Scale test returned: $X"

# Step 2: Loop from 0 to X and run ./main with parameters
for ((i=0; i<=X; i++)); do
  ./main --pow_size $X --N_Repeat 20 --pow_max_dist "$i"
done

python ../plot.py *.txt