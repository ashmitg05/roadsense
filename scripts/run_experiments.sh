#!/usr/bin/env bash
# Phase 2+3: baseline then one-variable-at-a-time experiments. Sequential (one GPU).
set -e
PY=.venv/Scripts/python
$PY -m src.data.oversample
for c in baseline exp_v8s exp_960 exp_aug exp_rfs; do
  [ -f results/$c.json ] || $PY -m src.train.train configs/train/$c.yaml > logs/$c.log 2>&1
done
until [ -d data/raw/United_States/train ]; do sleep 30; done
$PY -m src.data.convert --countries India Japan Czech United_States
$PY -m src.data.extra_countries
[ -f results/exp_multi.json ] || $PY -m src.train.train configs/train/exp_multi.yaml > logs/exp_multi.log 2>&1
echo ALL DONE
