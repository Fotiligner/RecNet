nodes=(
    ""
)

script=""
port=10000


n=4
first_gpu_id=0
n_num_per_process=2

for node in "${nodes[@]}"; do
    for ((i=0; i<${n}; i++))
    do
        # 初始化一个空字符串
        cur_gpu_id=""

        # 循环生成从 1 到 k 的整数
        for ((j=0; j<${n_num_per_process}; j++)); do
            gpu_id=$((first_gpu_id + i * n_num_per_process + j))
            if [ $j -eq 0 ]; then
                # 如果是最后一个数字，不加逗号
                cur_gpu_id+="${gpu_id}"
            else
                # 否则，加上逗号
                cur_gpu_id+=",${gpu_id}"
            fi
        done

        cur_port=$((port + i))
        # cur_gpu_id=$((first_gpu_id + i))

        command="source .bashrc && conda activate vllm-0.7.2 && source /opt/rh/devtoolset-11/enable && TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas bash ${script} ${cur_gpu_id} ${cur_port}"
        echo ${command}
        ssh "${node}" "${command}" &
    done
done