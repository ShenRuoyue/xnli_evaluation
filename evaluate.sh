# # source ~/.bashrc
# # . ~/.conda_init
# if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
#     . "/root/miniconda3/etc/profile.d/conda.sh"
# else
#     export PATH="/root/miniconda3/bin:$PATH"
# fi
# conda activate opencompass
# CUDA_VISIBLE_DEVICES=1 nohup python -u /app/evaluation/xnli/XNLI-1.0/evaluate.py > /app/evaluation/xnli/XNLI-1.0/outputs/eval_swallow-llama3-8b_ja_en_prompt.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u /app/evaluation/xnli/XNLI-1.0/evaluate.py > /app/evaluation/xnli/XNLI-1.0/outputs/eval_gemma2-2b-jpn-it_en.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2,3 nohup accelerate launch evaluate.py > /app/evaluation/xnli/XNLI-1.0/outputs/eval_gemma2-2b-it_de.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes 2  --main_process_port 23334 evaluate.py
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --main_process_port 23333 evaluate.py