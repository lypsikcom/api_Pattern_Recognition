#!/bin/bash

#:set ff=unix
#:set fileencoding=utf-8

export PATH=$PATH:/usr/local/python3/bin/
#if [ ! -d data  ];then
#  mkdir data
#else
#  echo dir data exist
#fi

if [ ! -d log  ];then
  mkdir log
else
  echo dir log exist
fi

#if [ ! -f console.log  ];then
#  touch console.log
#else
#  echo file console.log exist
#fi

# nohup python3 run.py >./log/console.log 2>>./log/console.log &
nohup python3 run.py >/dev/null 2>&1 &


