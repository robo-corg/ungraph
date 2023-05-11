#!/bin/bash
set -e

cargo build

mkdir -p logs

echo > logs/failures.csv

rm logs/*.txt

for M in $(fd '.onnx$' .)
do
    echo "$M"
    OUTPUT_NAME=$(echo "$M" | sed 's/.\///' | tr '/' '_')
    target/debug/ungraph "$M" 2>"logs/${OUTPUT_NAME}.stderr.txt" >"logs/${OUTPUT_NAME}.stdout.txt" || echo "$M" >> logs/failures.csv
done
