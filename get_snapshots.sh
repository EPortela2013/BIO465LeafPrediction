#!/bin/bash

wget http://edwardportela.com/bio465/snapshots.tar.xz && \
tar xf snapshots.tar.xz && \
rm snapshots.tar.xz && \
echo 'Successfully downloaded caffe models and snapshots'
