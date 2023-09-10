python main.py \
    --imsize 256 \
    --cropsize 256 \
    --type train \
    --batch_size 5 \
    --lr 1e-4 \
    --comment  Final_v9_adv_t2_no_adv  \
    --content_dir '../../dataset/MSCoCo' \
    --style_dir '../../dataset/wikiart' \
    --num_workers 16 \
    --max_iter 500001 \
    #--cencrop \