
iters=( 53000 )
for i in "${iters[@]}"
do

    CUDA_VISIBLE_DEVICES=1 python main.py \
        --type test \
        --batch_size 1 \
        --is_gan False \
        --comment  Final_v9_adv_t2 \
        --content_dir '../AesPA_Net/test_images/content_final' \
        --style_dir '../AesPA_Net/test_images/style_final' \
        --num_workers 16 \
        --test_iter $i \

done
