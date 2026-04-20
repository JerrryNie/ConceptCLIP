export CUDA_VISIBLE_DEVICES=1
cd fully_finetuning
python general_trainer.py --arch conceptclip --dataset-name 'SIIM-ACR' --batch-size 4 --accumulation-steps 16 --cache-dir 'finetune_cache'