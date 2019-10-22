#!/bin/bash

#Shuffle two parallel text files

file1=$1
file2=$2

mkfifo onerandom tworandom
tee onerandom tworandom < /dev/urandom > /dev/null &
shuf --random-source=onerandom $file1 > $file1.shuf &
shuf --random-source=tworandom $file2 > $file2.shuf &
wait

rm onerandom tworandom
