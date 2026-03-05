python training/pretrain_backbone_OCT2017.py --hflip --unweighted
python training/pretrain_backbone_OCT2017.py --hflip --unweighted --lr 0.0001 --eta_min 0.000001
python training/pretrain_backbone_OCT2017.py --hflip --unweighted --lr 0.01 --eta_min 0.0001
python training/pretrain_backbone_OCT2017.py --hflip --unweighted --lr 0.00001 --eta_min 0.0000001

python training/pretrain_backbone_OCT2017.py --hflip --unweighted --model L_16_imagenet1k --pretraining vit --batch_size 24
python training/pretrain_backbone_OCT2017.py --hflip --unweighted --model L_16_imagenet1k --pretraining vit --batch_size 24 --lr 0.0001 --eta_min 0.000001

python training/pretrain_backbone_OCT2017.py --hflip --unweighted --model vitb14_dino --pretraining dino --batch_size 16 --gpu 1
python training/pretrain_backbone_OCT2017.py --hflip --unweighted --model vitb14_dino --pretraining dino --batch_size 16 --lr 0.0001 --eta_min 0.000001 --gpu 1

#python training/pretrain_backbone_OCT2017.py --hflip --unweighted --model L_16_imagenet1k --pretraining vit --batch_size 24 --lr 0.01 --eta_min 0.0001
#python training/pretrain_backbone_OCT2017.py --hflip --unweighted --model L_16_imagenet1k --pretraining vit --batch_size 24 --lr 0.00001 --eta_min 0.0000001

python training/pretrain_backbone_OCT2017.py --hflip --unweighted --model L_16_imagenet1k --pretraining vit --batch_size 24 --lr 0.00001 --eta_min 0.0000001
python training/pretrain_backbone_OCT2017.py --hflip --unweighted --model vitb14_dino --pretraining dino --batch_size 16 --lr 0.00001 --eta_min 0.0000001
