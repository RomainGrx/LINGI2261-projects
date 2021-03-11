#!/bin/bash

Black=$'\e[1;30m'
Red=$'\e[1;31m'

function error(){
    echo "$Red ERROR :: $Black $1"
    exit
}

PYFILESWP=".swp.blocks"
SWP=".swp.benchmark"
SRC_INSTANCES="./instances"
INSTANCES=$(seq -f "%02g" 1 10)
OUT="results.csv"
HEADER="instance,time,moves,nodes explored,queue size"

cat blocks.py > $PYFILESWP # Copy the python file to a swap (snapshot)

# Check if the output csv already exists
if [[ -f $OUT ]]; then
	error "The output CSV $OUT alredy exists, please remove it before running the benchmark" 
fi

# Echo the header of the csv file 
echo $HEADER > $OUT

for instance in ${INSTANCES}; do
	echo "Instance $instance"
	timeout 60s python $PYFILESWP $SRC_INSTANCES/a$instance > $SWP
	if [[ $? -eq 124 ]]; then 
		echo "TO,TO,TO,TO" > $SWP
	fi
	echo -n "$instance," >> $OUT 
	cat $SWP >> $OUT
done

for swpfile in $SWP $PYFILESWP; do
	if [[ -f $swpfile ]]; then 
		rm $swpfile
	fi
done
