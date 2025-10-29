#!/bin/bash

declare -A NAME_DATE=(
    ["goto"]="1219"
    ["kawai"]="1115"
    ["matumoto"]="1128"
    ["takahashi"]="1220"
    ["takahashi_jr"]="0512"
    ["taniguchi"]="1107"
    ["yoshikura"]="1130"
    ["sato"]="1115"
    ["takamatu"]="1130"
    ["mori"]="0723"
    ["gosha"]="0807"
    ["asano"]="0714"
    ["togo"]="1107"
    ["takahashi_kazuya"]="0516"
    ["nakanishi"]="0407"
    ["miyano"]="0723"
    ["gobara"]="0512"
    ["kinoshita"]="0801"
    ["nakashimizu"]="0512"
    ["nishio"]="0513"
    ["kanda"]="0807"
    ["noda"]="0714"
    ["ikejima"]="0714"
    ["henmi"]="0304"
    ["henmi2"]="0304"
    ["kasahara"]="0304"
    ["maruyama"]="0304"
    ["miyabayashi"]="0304"
    ["patient1"]="1001"
    ["patient2"]="1001"
    ["patient3"]="1001"
    ["patient4"]="1001"
    ["patient5"]="1001"
    ["patient6"]="1001"
    ["patient7"]="1109"
    ["patient8"]="1109"
    ["patient9"]="1109"
    ["patient10"]="1109"
)

NAMES=(asano gosha goto ikejima kanda kawai matumoto nakashimizu nishio noda takahashi_jr taniguchi yoshikura patient2 patient3 patient4 patient5 patient6 patient7 patient8 patient9 patient10)

for NAME in "${NAMES[@]}"
do
  DATE=${NAME_DATE[$NAME]}
  python3 /mnt/ecg_project/src/dataset_creation/Make_dataset_resample_optimize.py --name "$NAME" --date "$DATE"
done
