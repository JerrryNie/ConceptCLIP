studies="BRACS-3"
ROOT_FEATURE="/path/to/feature"
for study in $studies
do
    CUDA_VISIBLE_DEVICES=0 nohup python main.py --model ABMIL \
                                                      --study $study \
                                                      --root ${ROOT_FEATURE}/${study} \
                                                      --feature conceptclip \
                                                      --csv_file ./dataset_csv/${study}.csv \
                                                      --num_epoch 50 \
                                                      --batch_size 1 \
                                                      --lr 2e-4 \
                                                      --tqdm



done

echo "All jobs have been submitted."