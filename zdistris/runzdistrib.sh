#!/bin/bash -xv

gcc -Wall -I/usr/local/include -L/usr/local/lib -o z-distribution z-distribution.c -lm
./z-distribution

rm z-distribution