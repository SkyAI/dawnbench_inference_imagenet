#!/bin/bash
g++ -O3 -std=c++11 infer.cpp -I/usr/local/cuda/include  -I./ -L/usr/local/cuda/lib64/ -L./ -lcudart $(pkg-config --cflags --libs opencv) -lengine -o inference
