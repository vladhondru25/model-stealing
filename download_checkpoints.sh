#!/bin/bash
set -e

gdown "https://drive.google.com/uc?id=1_UGjQRAGbwAvyBtTCQrXx1Cix_Vepjg6"
mkdir -p checkpoints
mv models.zip checkpoints
cd checkpoints
unzip models.zip
rm -fr models.zip
