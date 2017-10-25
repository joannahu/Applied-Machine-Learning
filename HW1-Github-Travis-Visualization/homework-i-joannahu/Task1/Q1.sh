#!/bin/bash

mkdir aml_hw1
cd aml_hw1
git init

for num in 1 2 3 4 5
do
touch $num
git add $num
git commit -m "commit file $num"
done

git branch feature HEAD~4
git checkout feature
for num in 6 7 8
do
touch $num
git add $num
git commit -m "commit file $num"
done

git checkout master
git rebase HEAD~2 --onto feature

git checkout HEAD@{11}
touch 9
git add 9
git commit -m "commit file 9"
git branch debug

git checkout master 7
git add 7
git commit --amend --no-edit


