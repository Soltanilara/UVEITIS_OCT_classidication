python training/train_OCT.py
python training/train_OCT.py --gpu 2 --lr 0.0001
python training/train_OCT.py --gpu 3 --lr 0.00001

python training/train_OCT.py --model resnet50w5 --gpu 0 --batch_size 16 --num_epochs 20

python training/train_OCT.py --model B_16_imagenet1k --pretraining vit --gpu 3

eval:
python training/train_OCT.py --mode eval
python training/train_OCT.py --mode eval --output_path './output (split_1)' --csvpath split_1
python training/train_OCT.py --mode eval --lr 0.0001 --output_path './output (split_1)' --csvpath split_1

python training/train_OCT.py --model resnet50w5 --gpu 2 --batch_size 16 --num_epochs 20 --mode eval

python training/train_OCT.py --csvpath split_1 --output_path output_split_1
python training/train_OCT.py --csvpath split_2 --output_path output_split_2
python training/train_OCT.py --csvpath split_3 --output_path output_split_3

python training/train_OCT.py --csvpath split_4 --output_path output_split_4
python training/train_OCT.py --csvpath split_5 --output_path output_split_5
python training/train_OCT.py --csvpath split_6 --output_path output_split_6

python training/train_OCT.py --csvpath split_7 --output_path output_split_7 --gpu 1
python training/train_OCT.py --csvpath split_8 --output_path output_split_8 --gpu 1
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --gpu 1

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --gpu 1
python training/train_OCT.py --csvpath split_11 --output_path output_split_11 --gpu 1
python training/train_OCT.py --csvpath split_12 --output_path output_split_12 --gpu 1


python training/train_OCT.py --csvpath split_1 --output_path output_split_1 --lr 0.00001
python training/train_OCT.py --csvpath split_2 --output_path output_split_2 --lr 0.00001
python training/train_OCT.py --csvpath split_3 --output_path output_split_3 --lr 0.00001

python training/train_OCT.py --csvpath split_1 --output_path output_split_1 --batch_size 32
python training/train_OCT.py --csvpath split_4 --output_path output_split_4 --lr 0.00001
python training/train_OCT.py --csvpath split_5 --output_path output_split_5 --lr 0.00001
python training/train_OCT.py --csvpath split_6 --output_path output_split_6 --lr 0.00001

python training/train_OCT.py --csvpath split_1 --output_path output_split_1 --lr 0.0001 --gpu 1
python training/train_OCT.py --csvpath split_7 --output_path output_split_7 --gpu 1 --lr 0.00001
python training/train_OCT.py --csvpath split_8 --output_path output_split_8 --gpu 1 --lr 0.00001
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --gpu 1 --lr 0.00001

python training/train_OCT.py --csvpath split_1 --output_path output_split_1 --lr 0.00001 --gpu 1
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --gpu 1 --lr 0.00001
python training/train_OCT.py --csvpath split_11 --output_path output_split_11 --gpu 1 --lr 0.00001
python training/train_OCT.py --csvpath split_12 --output_path output_split_12 --gpu 1 --lr 0.00001

----------------------------------------------------------

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --gpu 1
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --gpu 1

python training/train_OCT.py --csvpath split_8 --output_path output_split_8 --gpu 1
python training/train_OCT.py --csvpath split_7 --output_path output_split_7 --gpu 1

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --model B_16_imagenet1k --pretraining vit
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --model B_16_imagenet1k --pretraining vit

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --model L_16_imagenet1k --pretraining vit --batch_size 24
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --model L_16_imagenet1k --pretraining vit --batch_size 24

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --gpu 1 --protocol scratch --lr 0.001 --eta_min 0.00001
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --gpu 1 --protocol scratch --lr 0.01 --eta_min 0.0001
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --gpu 1 --lr 0.001 --eta_min 0.00001

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --elastic --brightness --contrast --gnoise
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --protocol scratch --hflip --elastic --brightness --contrast --gnoise
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --elastic --brightness --contrast --gnoise
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --protocol scratch --hflip --elastic --brightness --contrast --gnoise

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --elastic

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --brightness --contrast --gpu 1
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --gnoise --gpu 1

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --protocol scratch --hflip
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --protocol scratch --elastic
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --protocol scratch --brightness --contrast
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --protocol scratch --gnoise

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --model vitb14_dino --pretraining dino --batch_size 16
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --model vitb14_dino --pretraining dino --lr 0.0001 --eta_min 0.000001 --batch_size 16
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --model vitb14_dino --pretraining dino --gpu 1 --batch_size 16
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --model vitb14_dino --pretraining dino --gpu 1 --lr 0.0001 --eta_min 0.000001 --batch_size 16

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --protocol scratch --hflip --loss focal
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --protocol scratch --hflip --loss focal --lr 0.0001 --eta_min 0.000001
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --protocol scratch --hflip --loss focal --lr 0.01 --eta_min 0.0001

python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --protocol scratch --hflip --loss focal --gpu 1
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --protocol scratch --hflip --loss focal --lr 0.0001 --eta_min 0.000001 --gpu 1
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --protocol scratch --hflip --loss focal --lr 0.01 --eta_min 0.0001 --gpu 1

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --protocol scratch --hflip --loss focal --gamma 1
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --protocol scratch --hflip --loss focal --lr 0.0001 --eta_min 0.000001 --gamma 1
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --protocol scratch --hflip --loss focal --lr 0.01 --eta_min 0.0001 --gamma 1

python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --protocol scratch --hflip --loss focal --gamma 1 --gpu 1
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --protocol scratch --hflip --loss focal --lr 0.0001 --eta_min 0.000001 --gamma 1 --gpu 1
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --protocol scratch --hflip --loss focal --lr 0.01 --eta_min 0.0001 --gamma 1 --gpu 1

# latent space evals
python evaluation/eval_OCT_latent.py --csvpath split_10 --output_path output_split_10 --protocol scratch --hflip --loss focal --gamma 1 --gpu 1
python evaluation/eval_OCT_latent.py --csvpath split_10 --output_path output_split_10 --protocol scratch --hflip --loss CE

# graded training
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --lr 0.1 --eta_min 0.001
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --lr 0.01 --eta_min 0.0001

python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --lr 0.001 --eta_min 0.00001
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --lr 0.0001 --eta_min 0.000001

python training/train_OCT_graded.py --csvpath split_9 --output_path output_graded_split_9 --hflip --lr 0.1 --eta_min 0.001 --gpu 1
python training/train_OCT_graded.py --csvpath split_9 --output_path output_graded_split_9 --hflip --lr 0.01 --eta_min 0.0001 --gpu 1

python training/train_OCT_graded.py --csvpath split_9 --output_path output_graded_split_9 --hflip --lr 0.001 --eta_min 0.00001 --gpu 1
python training/train_OCT_graded.py --csvpath split_9 --output_path output_graded_split_9 --hflip --lr 0.0001 --eta_min 0.000001 --gpu 1

python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --lr 0.001 --eta_min 0.00001 --p 2
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --lr 0.001 --eta_min 0.00001 --p 2 --protocol finetune

python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --lr 0.001 --eta_min 0.00001 --unweighted
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --lr 0.001 --eta_min 0.00001 --unweighted --protocol finetune

python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --lr 0.001 --eta_min 0.00001 --loss CE --gpu 1
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --lr 0.001 --eta_min 0.00001 --unweighted --loss CE --gpu 1

python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --lr 0.001 --eta_min 0.00001 --p 5 --gpu 1
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --lr 0.001 --eta_min 0.00001 --p 10 --gpu 1

python training/train_OCT_graded.py --csvpath split_9 --output_path output_graded_split_9 --hflip --unweighted --protocol finetune --gpu 1

# Unweighted binary
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --lr 1 --eta_min 0.01
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --lr 0.000001 --eta_min 0.00000001

python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --unweighted --gpu 1

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --model vgg19_bn --pretraining supervised
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --model vgg19_bn --pretraining supervised --lr 0.0001 --eta_min 0.000001

python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --model vgg19_bn --pretraining supervised --gpu 1
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --model vgg19_bn --pretraining supervised --lr 0.0001 --eta_min 0.000001 --gpu 1

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --model vgg19_bn --pretraining supervised --lr 0.00001 --eta_min 0.0000001
python training/train_OCT_graded.py --csvpath split_8 --output_path output_graded_split_8 --hflip --unweighted
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --model vgg19_bn --pretraining supervised --lr 0.00001 --eta_min 0.0000001 --gpu 1
python training/train_OCT_graded.py --csvpath split_7 --output_path output_graded_split_7 --hflip --unweighted --gpu 1


python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --model resnet18 --pretraining supervised
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --model densenet121 --pretraining supervised --batch_size 48
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --model densenet201 --pretraining supervised --batch_size 32

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --model densenet201 --protocol scratch --batch_size 32
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --model densenet121 --protocol scratch --batch_size 48
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --model resnet18 --protocol scratch

python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --model resnet18 --pretraining supervised --gpu 1
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --model densenet121 --pretraining supervised --gpu 1 --batch_size 48
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --model densenet201 --pretraining supervised --gpu 1 --batch_size 32

python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --model densenet201 --protocol scratch --gpu 1 --batch_size 32
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --model densenet121 --protocol scratch --gpu 1 --batch_size 48
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --model resnet18 --protocol scratch --gpu 1

python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --mimo
python training/train_OCT_graded.py --csvpath split_8 --output_path output_graded_split_8 --hflip --unweighted --mimo

python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --mose
python training/train_OCT_graded.py --csvpath split_8 --output_path output_graded_split_8 --hflip --unweighted --mose

python training/train_OCT_graded.py --csvpath split_9 --output_path output_graded_split_9 --hflip --unweighted --mimo --gpu 1
python training/train_OCT_graded.py --csvpath split_7 --output_path output_graded_split_7 --hflip --unweighted --mimo --gpu 1

python training/train_OCT_graded.py --csvpath split_9 --output_path output_graded_split_9 --hflip --unweighted --mose --gpu 1
python training/train_OCT_graded.py --csvpath split_7 --output_path output_graded_split_7 --hflip --unweighted --mose --gpu 1

# AE runs:
python training/train_OCT_AE.py --csvpath split_9 --output_path output_AE_split_9 --hflip --beta 1.0 --batch_size 32 --mode eval
python training/train_OCT_AE.py --csvpath split_9 --output_path output_AE_split_9 --hflip --beta 0.1 --batch_size 32 --mode eval
python training/train_OCT_AE.py --csvpath split_8 --output_path output_AE_split_8 --hflip --beta 1.0 --batch_size 32 --mode eval
python training/train_OCT_AE.py --csvpath split_8 --output_path output_AE_split_8 --hflip --beta 0.1 --batch_size 32 --mode eval

python training/train_OCT_AE.py --csvpath split_9 --output_path output_AE_split_9 --hflip --beta 10.0 --batch_size 32 --mode eval
python training/train_OCT_AE.py --csvpath split_9 --output_path output_AE_split_9 --hflip --beta 0.01 --batch_size 32 --mode eval
python training/train_OCT_AE.py --csvpath split_8 --output_path output_AE_split_8 --hflip --beta 10.0 --batch_size 32 --mode eval
python training/train_OCT_AE.py --csvpath split_8 --output_path output_AE_split_8 --hflip --beta 0.01 --batch_size 32 --mode eval

python training/train_OCT_AE.py --csvpath split_9 --output_path output_AE_split_9 --hflip --protocol scratch --beta 1.0 --gpu 1 --batch_size 32 --mode eval --gpu 1
python training/train_OCT_AE.py --csvpath split_9 --output_path output_AE_split_9 --hflip --protocol scratch --beta 0.1 --gpu 1 --batch_size 32 --mode eval --gpu 1
python training/train_OCT_AE.py --csvpath split_8 --output_path output_AE_split_8 --hflip --protocol scratch --beta 1.0 --gpu 1 --batch_size 32 --mode eval --gpu 1
python training/train_OCT_AE.py --csvpath split_8 --output_path output_AE_split_8 --hflip --protocol scratch --beta 0.1 --gpu 1 --batch_size 32 --mode eval --gpu 1

python training/train_OCT_AE.py --csvpath split_9 --output_path output_AE_split_9 --hflip --protocol scratch --beta 10.0 --gpu 1 --batch_size 32 --mode eval --gpu 1
python training/train_OCT_AE.py --csvpath split_9 --output_path output_AE_split_9 --hflip --protocol scratch --beta 0.01 --gpu 1 --batch_size 32 --mode eval --gpu 1
python training/train_OCT_AE.py --csvpath split_8 --output_path output_AE_split_8 --hflip --protocol scratch --beta 10.0 --gpu 1 --batch_size 32 --mode eval --gpu 1
python training/train_OCT_AE.py --csvpath split_8 --output_path output_AE_split_8 --hflip --protocol scratch --beta 0.01 --gpu 1 --batch_size 32 --mode eval --gpu 1

# Graded Training with Transformer
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --batch_size 24
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --mimo --model L_16_imagenet1k --pretraining vit --batch_size 24
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --mose --model L_16_imagenet1k --pretraining vit --batch_size 24

python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --gpu 1 --batch_size 24
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --mimo --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --gpu 1 --batch_size 24
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --mose --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --gpu 1 --batch_size 24

python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --batch_size 24 --loss CE
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --mimo --model L_16_imagenet1k --pretraining vit --batch_size 24 --loss CE
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --mose --model L_16_imagenet1k --pretraining vit --batch_size 24 --loss CE

python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --gpu 1 --batch_size 24 --loss CE
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --mimo --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --gpu 1 --batch_size 24 --loss CE
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --mose --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --gpu 1 --batch_size 24 --loss CE

python training/train_OCT_AE.py --csvpath split_3 --output_path output_AE_split_3 --hflip --beta 1.0 --batch_size 32
python training/train_OCT_AE.py --csvpath split_3 --output_path output_AE_split_3 --hflip --beta 10.0 --batch_size 32

python training/train_OCT_AE.py --csvpath split_3 --output_path output_AE_split_3 --hflip --beta 0.1 --batch_size 32
python training/train_OCT_AE.py --csvpath split_3 --output_path output_AE_split_3 --hflip --beta 0.01 --batch_size 32

python training/train_OCT_AE.py --csvpath split_6 --output_path output_AE_split_6 --hflip --beta 1.0 --batch_size 32 --gpu 1
python training/train_OCT_AE.py --csvpath split_6 --output_path output_AE_split_6 --hflip --beta 10.0 --batch_size 32 --gpu 1

python training/train_OCT_AE.py --csvpath split_6 --output_path output_AE_split_6 --hflip --beta 0.1 --batch_size 32 --gpu 1
python training/train_OCT_AE.py --csvpath split_6 --output_path output_AE_split_6 --hflip --beta 0.01 --batch_size 32 --gpu 1

# OCT2017 ----------------------------------------------------------------------------------
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --pretraining oct --gpu 1
python training/train_OCT.py --csvpath split_6 --output_path output_split_6 --hflip --pretraining oct --gpu 1
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --pretraining oct --gpu 1
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --pretraining oct --lr 0.0001 --eta_min 0.000001 --gpu 1
python training/train_OCT.py --csvpath split_6 --output_path output_split_6 --hflip --pretraining oct --lr 0.0001 --eta_min 0.000001 --gpu 1
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --pretraining oct --lr 0.0001 --eta_min 0.000001 --gpu 1

python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --pretraining oct --lr 0.0001 --eta_min 0.000001 --model L_16_imagenet1k --batch_size 24
python training/train_OCT.py --csvpath split_6 --output_path output_split_6 --hflip --pretraining oct --lr 0.0001 --eta_min 0.000001 --model L_16_imagenet1k --batch_size 24
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --pretraining oct --lr 0.0001 --eta_min 0.000001 --model L_16_imagenet1k --batch_size 24

python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --pretraining oct --lr 0.0001 --eta_min 0.000001 --model vitb14_dino --batch_size 16 --gpu 1
python training/train_OCT.py --csvpath split_6 --output_path output_split_6 --hflip --pretraining oct --lr 0.0001 --eta_min 0.000001 --model vitb14_dino --batch_size 16 --gpu 1
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --pretraining oct --lr 0.0001 --eta_min 0.000001 --model vitb14_dino --batch_size 16 --gpu 1

#Dataset 01032025

# Binary weighted and unweighted, with and without oct pretraining
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --pretraining oct
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --pretraining oct --unweighted

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --unweighted
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --pretraining oct
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --pretraining oct --unweighted

python training/train_OCT.py --csvpath split_3 --output_path output_split_3 --hflip --gpu 1
python training/train_OCT.py --csvpath split_2 --output_path output_split_2 --hflip --gpu 1
python training/train_OCT.py --csvpath split_3 --output_path output_split_3 --hflip --pretraining oct --gpu 1
python training/train_OCT.py --csvpath split_3 --output_path output_split_3 --hflip --pretraining oct --unweighted --gpu 1

python training/train_OCT.py --csvpath split_3 --output_path output_split_3 --hflip --unweighted --gpu 1
python training/train_OCT.py --csvpath split_2 --output_path output_split_2 --hflip --unweighted --gpu 1
python training/train_OCT.py --csvpath split_2 --output_path output_split_2 --hflip --pretraining oct --gpu 1
python training/train_OCT.py --csvpath split_2 --output_path output_split_2 --hflip --pretraining oct --unweighted --gpu 1

# Last epoch evals and 50 epoch evals
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --mode eval --checkpoint checkpoint_epoch_100.pt
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --mode eval --checkpoint checkpoint_epoch_100.pt
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_100.pt
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_100.pt
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --mode eval --checkpoint checkpoint_epoch_50.pt
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --mode eval --checkpoint checkpoint_epoch_50.pt
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_50.pt
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_50.pt

python training/train_OCT.py --csvpath split_3 --output_path output_split_3 --hflip --gpu 1 --mode eval --checkpoint checkpoint_epoch_100.pt
python training/train_OCT.py --csvpath split_2 --output_path output_split_2 --hflip --gpu 1 --mode eval --checkpoint checkpoint_epoch_100.pt
python training/train_OCT.py --csvpath split_3 --output_path output_split_3 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_100.pt
python training/train_OCT.py --csvpath split_2 --output_path output_split_2 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_100.pt
python training/train_OCT.py --csvpath split_3 --output_path output_split_3 --hflip --gpu 1 --mode eval --checkpoint checkpoint_epoch_50.pt
python training/train_OCT.py --csvpath split_2 --output_path output_split_2 --hflip --gpu 1 --mode eval --checkpoint checkpoint_epoch_50.pt
python training/train_OCT.py --csvpath split_3 --output_path output_split_3 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_50.pt
python training/train_OCT.py --csvpath split_2 --output_path output_split_2 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_50.pt


# OCT evals 100 and 50 epoch checkpoints
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --mode eval --checkpoint checkpoint_epoch_100.pt --pretraining oct
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --mode eval --checkpoint checkpoint_epoch_100.pt --pretraining oct
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_100.pt --pretraining oct
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_100.pt --pretraining oct
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --mode eval --checkpoint checkpoint_epoch_50.pt --pretraining oct
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --mode eval --checkpoint checkpoint_epoch_50.pt --pretraining oct
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_50.pt --pretraining oct
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_50.pt --pretraining oct

python training/train_OCT.py --csvpath split_3 --output_path output_split_3 --hflip --gpu 1 --mode eval --checkpoint checkpoint_epoch_100.pt --pretraining oct
python training/train_OCT.py --csvpath split_2 --output_path output_split_2 --hflip --gpu 1 --mode eval --checkpoint checkpoint_epoch_100.pt --pretraining oct
python training/train_OCT.py --csvpath split_3 --output_path output_split_3 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_100.pt --pretraining oct
python training/train_OCT.py --csvpath split_2 --output_path output_split_2 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_100.pt --pretraining oct
python training/train_OCT.py --csvpath split_3 --output_path output_split_3 --hflip --gpu 1 --mode eval --checkpoint checkpoint_epoch_50.pt --pretraining oct
python training/train_OCT.py --csvpath split_2 --output_path output_split_2 --hflip --gpu 1 --mode eval --checkpoint checkpoint_epoch_50.pt --pretraining oct
python training/train_OCT.py --csvpath split_3 --output_path output_split_3 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_50.pt --pretraining oct
python training/train_OCT.py --csvpath split_2 --output_path output_split_2 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_50.pt --pretraining oct

python training/train_OCT.py --csvpath split_1 --output_path output_split_1 --hflip --unweighted
python training/train_OCT.py --csvpath split_4 --output_path output_split_4 --hflip --unweighted

python training/train_OCT.py --csvpath split_5 --output_path output_split_5 --hflip --unweighted
python training/train_OCT.py --csvpath split_6 --output_path output_split_6 --hflip --unweighted

python training/train_OCT.py --csvpath split_7 --output_path output_split_7 --hflip --unweighted --gpu 1
python training/train_OCT.py --csvpath split_8 --output_path output_split_8 --hflip --unweighted --gpu 1

python training/train_OCT.py --csvpath split_11 --output_path output_split_11 --hflip --unweighted --gpu 1
python training/train_OCT.py --csvpath split_12 --output_path output_split_12 --hflip --unweighted --gpu 1

python training/train_OCT.py --csvpath split_1 --output_path output_split_1 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_50.pt
python training/train_OCT.py --csvpath split_4 --output_path output_split_4 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_50.pt
python training/train_OCT.py --csvpath split_1 --output_path output_split_1 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_100.pt
python training/train_OCT.py --csvpath split_4 --output_path output_split_4 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_100.pt

python training/train_OCT.py --csvpath split_5 --output_path output_split_5 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_50.pt
python training/train_OCT.py --csvpath split_6 --output_path output_split_6 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_50.pt
python training/train_OCT.py --csvpath split_5 --output_path output_split_5 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_100.pt
python training/train_OCT.py --csvpath split_6 --output_path output_split_6 --hflip --unweighted --mode eval --checkpoint checkpoint_epoch_100.pt

python training/train_OCT.py --csvpath split_7 --output_path output_split_7 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_50.pt
python training/train_OCT.py --csvpath split_8 --output_path output_split_8 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_50.pt
python training/train_OCT.py --csvpath split_7 --output_path output_split_7 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_100.pt
python training/train_OCT.py --csvpath split_8 --output_path output_split_8 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_100.pt

python training/train_OCT.py --csvpath split_11 --output_path output_split_11 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_50.pt
python training/train_OCT.py --csvpath split_12 --output_path output_split_12 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_50.pt
python training/train_OCT.py --csvpath split_11 --output_path output_split_11 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_100.pt
python training/train_OCT.py --csvpath split_12 --output_path output_split_12 --hflip --unweighted --gpu 1 --mode eval --checkpoint checkpoint_epoch_100.pt

# AE runs:
python training/train_OCT_AE.py --csvpath split_4 --output_path output_AE_split_4 --hflip --beta 1.0 --unweighted --batch_size 32
python training/train_OCT_AE.py --csvpath split_10 --output_path output_AE_split_10 --hflip --beta 1.0 --unweighted --batch_size 32
python training/train_OCT_AE.py --csvpath split_6 --output_path output_AE_split_6 --hflip --beta 1.0 --unweighted --batch_size 32 --gpu 1
python training/train_OCT_AE.py --csvpath split_7 --output_path output_AE_split_7 --hflip --beta 1.0 --unweighted --batch_size 32 --gpu 1


python training/train_OCT.py --csvpath split_13 --output_path output_split_13 --hflip --unweighted
python training/train_OCT.py --csvpath split_17 --output_path output_split_17 --hflip --unweighted

python training/train_OCT.py --csvpath split_14 --output_path output_split_14 --hflip --unweighted
python training/train_OCT.py --csvpath split_18 --output_path output_split_18 --hflip --unweighted

python training/train_OCT.py --csvpath split_15 --output_path output_split_15 --hflip --unweighted --gpu 1
python training/train_OCT.py --csvpath split_19 --output_path output_split_19 --hflip --unweighted --gpu 1

python training/train_OCT.py --csvpath split_16 --output_path output_split_16 --hflip --unweighted --gpu 1
python training/train_OCT.py --csvpath split_20 --output_path output_split_20 --hflip --unweighted --gpu 1

#Graded
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --loss CE
python training/train_OCT_graded.py --csvpath split_10 --output_path output_graded_split_10 --mose --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --loss CE

python training/train_OCT_graded.py --csvpath split_9 --output_path output_graded_split_9 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --gpu 1 --batch_size 24 --loss CE
python training/train_OCT_graded.py --csvpath split_9 --output_path output_graded_split_9 --mose --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --gpu 1 --batch_size 24 --loss CE


# Binary with ViT_L19_in1k
python training/train_OCT.py --csvpath split_4 --output_path output_split_4 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24

python training/train_OCT.py --csvpath split_6 --output_path output_split_6 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --gpu 1
python training/train_OCT.py --csvpath split_7 --output_path output_split_7 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --gpu 1

python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24
python training/train_OCT.py --csvpath split_17 --output_path output_split_17 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24
python training/train_OCT.py --csvpath split_3 --output_path output_split_3 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24
python training/train_OCT.py --csvpath split_18 --output_path output_split_18 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24
python training/train_OCT.py --csvpath split_19 --output_path output_split_19 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24

python training/train_OCT.py --csvpath split_15 --output_path output_split_15 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --gpu 1
python training/train_OCT.py --csvpath split_16 --output_path output_split_16 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --gpu 1
python training/train_OCT.py --csvpath split_14 --output_path output_split_14 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --gpu 1
python training/train_OCT.py --csvpath split_13 --output_path output_split_13 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --gpu 1

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --weightedSampling
python training/train_OCT.py --csvpath split_7 --output_path output_split_7 --hflip --unweighted --weightedSampling

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --mixup
python training/train_OCT.py --csvpath split_7 --output_path output_split_7 --hflip --unweighted --mixup

python training/train_OCT.py --csvpath split_4 --output_path output_split_4 --hflip --unweighted --weightedSampling --gpu 1
python training/train_OCT.py --csvpath split_7 --output_path output_split_7 --hflip --unweighted --mixup --mixup_alpha 0.2 --mixup_beta 0.2 --gpu 1

python training/train_OCT.py --csvpath split_4 --output_path output_split_4 --hflip --unweighted --mixup --gpu 1
python training/train_OCT.py --csvpath split_7 --output_path output_split_7 --hflip --unweighted --weightedSampling --mixup --gpu 1

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --mixup --weightedSampling
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --mixup --weightedSampling --T_multi 2
python training/train_OCT.py --csvpath split_4 --output_path output_split_4 --hflip --unweighted --mixup --weightedSampling --gpu 1
python training/train_OCT.py --csvpath split_4 --output_path output_split_4 --hflip --unweighted --mixup --weightedSampling --gpu 1 --T_multi 2

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --mixup --T_multi 2
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --mixup --T_multi 2 --lr 0.0001 --eta_min 0.000001

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --mixup --T_multi 2
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --mixup --T_multi 2 --lr 0.0001 --eta_min 0.000001

python training/train_OCT.py --csvpath split_4 --output_path output_split_4 --hflip --unweighted --mixup --gpu 1 --T_multi 2
python training/train_OCT.py --csvpath split_4 --output_path output_split_4 --hflip --unweighted --mixup --gpu 1 --T_multi 2 --lr 0.0001 --eta_min 0.000001

python training/train_OCT.py --csvpath split_4 --output_path output_split_4 --hflip --mixup --gpu 1 --T_multi 2
python training/train_OCT.py --csvpath split_4 --output_path output_split_4 --hflip --mixup --gpu 1 --T_multi 2 --lr 0.0001 --eta_min 0.000001

python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --unweighted --mixup
python training/train_OCT.py --csvpath split_15 --output_path output_split_15 --hflip --unweighted --mixup
python training/train_OCT.py --csvpath split_6 --output_path output_split_6 --hflip --unweighted --mixup --gpu 1
python training/train_OCT.py --csvpath split_17 --output_path output_split_17 --hflip --unweighted --mixup --gpu 1

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --mixup
python training/train_OCT.py --csvpath split_15 --output_path output_split_15 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --mixup --gpu 1

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --mixup --decay 0.0001
python training/train_OCT.py --csvpath split_9 --output_path output_split_9 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --mixup

python training/train_OCT.py --csvpath split_15 --output_path output_split_15 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --mixup --gpu 1 --decay 0.0001
python training/train_OCT.py --csvpath split_4 --output_path output_split_4 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --mixup --gpu 1

python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --mixup --decay 0.001
python training/train_OCT.py --csvpath split_10 --output_path output_split_10 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --mixup --decay 0.1

python training/train_OCT.py --csvpath split_15 --output_path output_split_15 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --mixup --gpu 1 --decay 0.001
python training/train_OCT.py --csvpath split_15 --output_path output_split_15 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --mixup --gpu 1 --decay 0.1

python training/train_OCT.py --csvpath split_15 --output_path output_split_15 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --mixup --harmonic --gpu 1
python training/train_OCT.py --csvpath split_15 --output_path output_split_15 --hflip --unweighted --model L_16_imagenet1k --pretraining vit --lr 0.0001 --eta_min 0.000001 --batch_size 24 --mixup --harmonic --n_harmax 1

python training/train_OCT.py --csvpath split_15 --output_path output_split_15 --hflip --unweighted --gpu 1 --harmonic --n_harmax 1
python training/train_OCT.py --csvpath split_15 --output_path output_split_15 --hflip --unweighted --gpu 1 --harmonic --n_harmax 45
