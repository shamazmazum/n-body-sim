#!/bin/sh

if [ -f "$1" ]; then
    defsize=$2
    size=${2:-20}
    gnuplot << EOF
size = $size

set terminal png size 1920,1080 enhanced font "Helvetica,20"
set xrange [-size: size]
set yrange [-size: size]

set output "$1.png"
plot "$1" with dots
EOF
fi
