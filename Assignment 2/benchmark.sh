#!/bin/bash

Black=$'\e[1;30m'
Red=$'\e[1;31m'

function error(){
    echo "$Red ERROR :: $Black $1"
    exit
}

SRC_INSTANCES="./instances"
INSTANCES=$(seq -f "%02g" 1 10)
OUT="results.csv"
SWP=".swp.benchmark"
HEADER="instance,time,moves,nodes explored,queue size"

if [[ -f $OUT ]]; then
	error "The output CSV $OUT alredy exists, please remove it before running the benchmark" 
fi

echo $HEADER > $OUT

for instance in ${INSTANCES}; do
	echo "Instance $instance"
	timeout 60s python blocks.py $SRC_INSTANCES/a$instance > $SWP
	if [[ $? -eq 124 ]]; then 
		echo "TO,TO,TO,TO" > $SWP
	fi
	echo -n "$instance," >> $OUT 
	cat $SWP >> $OUT
done

if [[ -f $SWP ]]; then 
	rm $SWP
fi
