# Predicting the Sum of Long Numbers with Language Models

The aim of the task was to adapt language model to perform addition of two long numbers.  To do it, I used a pre-trained T5-base model, as  it was described in the paper [Nogueira, Jiang, Lin "Investigating the Limitations of Transformers with Simple Arithmetic Tasks", 2021](https://arxiv.org/abs/2102.13019). The code was mainly taken from the [corresponding](https://github.com/castorini/transformers-arithmetic/tree/main) repository.

The model is able to perform the addition of two 50-digit numbers with 99.6% accuracy.

# Data 
The task is to sum up to long numbers. An example input is "103, 54", and output is "158".  We would use ”10e-based” (when 103 is presented like 1 10e2 0 10 e1 3 10e0) number representation as it achieved the best accuracy in the experiments  conducted in the paper.

To generate  datasets, we set the maximum number of digits $D$, choose $d$ from $[2; D]$  and create two numbers from  $[10^{d-1}, 10^d - 1]$, and compute the answer as sum of this numbers. With this method we have equal proportion of $d$-digit numbers.

# How to train

To install dependencies, use  `pip install -r requirements.txt`
To avoid overfitting, we check accuracy on validation set after every epoch, and do checkpoint.

To train the model on the task of two 50-digits numbers addition, you can use this command:

    python train.py \
    --output_dir=. \
    --model_name_or_path=t5-base \
    --train_size=20000 \
    --val_size=1000 \
    --min_digits_train=2 \
    --max_digits_train=50 \
    --seed=1 \
    --train_batch_size=4 \
    --accumulate_grad_batches=32 \
    --val_batch_size=32 \
    --max_seq_length=512 \
    --num_workers=4 \
    --gpus=1 \
    --optimizer=AdamW \
    --lr=3e-4 \
    --weight_decay=5e-5 \
    --scheduler=StepLR \
    --t_0=2 \
    --t_mult=2 \
    --gamma=1.0 \
    --step_size=1000 \
    --max_epochs=10 \
    --step_size=1000 \
    --max_epochs=10 \
    --gradient_clip_val=1.0

## How to test

To test the model, you can download the best [checkpoint](https://drive.google.com/file/d/1opsyav4gujFTBTbvB6X0lDJAsZMBj2t8/view?usp=share_link) that I got during training,  and put in the same directory.

Run this command to test model on the 1000-examples dataset with checkpoint mentioned above.

    python test.py \
    --output_dir=. \
    --checkpoint_name='epoch=4-val_exact_match=0.9960.ckpt' \
    --seed=1 \
    --test_size=1000 \
    --num_workers=4 \
    --min_digits_test=2 \
    --max_digits_test=50 \
    --gpus=1

## How to predict

If you want to make a single prediction, run this command:

    python predict.py \
    --output_dir=. \
    --first_number=4 \
    --second_number=5 \
    --checkpoint_name='epoch=4-val_exact_match=0.9960.ckpt' \
    --gpus=1

## Extrapolation

We investigated how model perform extrapolation task. The results are shown below.

<img width="814" alt="Снимок экрана 2023-05-05 в 09 29 10" src="https://user-images.githubusercontent.com/114249608/236399918-7dc2a376-32b3-4ac7-b0f6-712afe0dc6a9.png">

It makes sense to add up numbers up to 65-digits length.
