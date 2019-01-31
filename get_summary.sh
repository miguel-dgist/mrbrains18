#!/bin/sh

RUNS=$(ls train/)

pwd > "summary.txt"
echo "
" >> "summary.txt"

for run in $RUNS
do
  SUMMARIES=$(ls "train/${run}/summaries/")
  echo "$run
    " >> "summary.txt"
  for summary in $SUMMARIES
  do
    echo "$summary" >> "summary.txt"
    cat "train/${run}/summaries/${summary}" | grep "average" \
      >> "summary.txt"
    echo "" >> "summary.txt"
  done
done

