#!/usr/bin/env bash
trap "exit" INT
base_command="python train_digits.py --net digit -ni 10000 -vi 250 -bs 32 -lr 0.0001 -sc mnist10k -chs 3 "

settings=(
" "                          # baseline
" --grey "                   # greyscale
" -jitter "                  # colorjitter
" -LoG "                     # BandPass
" -ma "                      # MultiAug

# RC_img p=0.5, with consistency loss and lambda=5
" -ks 1 3 5 7 -rc -idp 0.5 -cl -clw 5 -db kaiming_normal -vwr -nv 10 "

# RC_mix, with consistency loss and lambda=10
" -ks 1 3 5 7 -rc -mix -cl -clw 10 -db kaiming_normal -vwr -nv 10 "

# MultiAug + RC_mix with consistency loss and lambda=10
" -ma -ks 1 3 5 7 -rc -mix -cl -clw 10 -db kaiming_normal -vwr -nv 10 "
)

#rand_seeds=(1 2 3 4 5)
rand_seeds=(1  )

for rs in ${rand_seeds[@]}
    do
        for setting in "${settings[@]}"
        do
            $base_command -g "$1" -rs $rs $setting # run training and testing
            $base_command -g "$1" -rs $rs $setting -crpt -test # testing on MNIST-C
        done
    done

