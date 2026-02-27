set -ex

ckpt=$1
data=$2

step=15
# for budget in 7 8
# do
#     torchrun --nproc_per_node=4 main.py --data-folder ${data} --bsize 128 --writer-log logs --num_workers 16 --img-size 512 --resume --test-only --metric_save_dir result_metrics/shortcut --use_shortcut --shortcut_path ${ckpt} --bottleneck_ratio 2 --budget ${budget} --step ${step}
# done

# step=32
# for budget in 8 9 12
# do
#     torchrun --nproc_per_node=4 main.py --data-folder ${data} --bsize 128 --writer-log logs --num_workers 16 --img-size 512 --resume --test-only --metric_save_dir result_metrics/shortcut --use_shortcut --shortcut_path ${ckpt} --bottleneck_ratio 2 --budget ${budget} --step ${step}
# done

torchrun --nproc_per_node=4 main.py --data-folder ${data} --bsize 128 --writer-log logs --num_workers 16 --img-size 512 --resume --test-only --metric_save_dir result_metrics/shortcut --step ${step}
