#!/bin/bash

[ ! -d "./dataset/utkface/part1" ] && mkdir -p ./dataset/utkface/part1
[ ! -d "./dataset/utkface/part2" ] && mkdir -p ./dataset/utkface/part2
[ ! -d "./dataset/utkface/part3" ] && mkdir -p ./dataset/utkface/part3

echo "Extracting part 1"
tar xf utkface_1.tar.gz --directory=./dataset/utkface
echo "Extracting part 2"
tar xf utkface_2.tar.gz --directory=./dataset/utkface
echo "Extracting part 3"
tar xf utkface_3.tar.gz --directory=./dataset/utkface

echo "Moving files"
mv ./dataset/utkface/part1/*.jpg ./dataset/utkface
mv ./dataset/utkface/part2/*.jpg ./dataset/utkface
mv ./dataset/utkface/part3/*.jpg ./dataset/utkface

echo "Deleting extras"
rm -fr ./dataset/utkface/part1
rm -fr ./dataset/utkface/part2
rm -fr ./dataset/utkface/part3

rm ./dataset/utkface/*__*
cat alphas.txt | xargs rm
cat grays.txt | xargs rm