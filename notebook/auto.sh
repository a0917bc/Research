# -*- coding: utf-8 -*-
TWCC_CLI_CMD=/home/u1887834/.local/bin/twccli
LAYER=$1
DEVICES=$2
ID=$3
date
echo "1. 執行運算"
python 4GPU_test.py
echo "2. 開發型容器"
$TWCC_CLI_CMD rm ccs -f -s $ID
# 3933199, for experiment int3, int2跟int1

