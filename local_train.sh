export CUDA_HOME=/usr/local/cuda-11.7
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:./

gpus=(${CUDA_VISIBLE_DEVICES//,/ })
gpu_num=${#gpus[@]}
echo "number of gpus: "${gpu_num}

config=projects/configs/$1.py

if [ ${gpu_num} -gt 1 ]
then
    bash ./tools/dist_train.sh \
        --config=$config \
        # ${config} \
        ${gpu_num} \
        --work-dir=work_dirs/$1
else
    python ./tools/train.py \
        --config=$config 
fi
