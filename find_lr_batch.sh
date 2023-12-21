# Array met verschillende batchgroottes
batch_sizes=(5 10 15 20 25 30)

# Array met verschillende learningrates
learning_rates=(0.001 0.01 0.1)

# Loop over batchgroottes en learningrates
for batch_size in "${batch_sizes[@]}"
do
    for lr in "${learning_rates[@]}"
    do
        python3 train.py --wandb_entity=aarontenzing --wandb_project=labo_beeld --batch_size=$batch_size --lr=$lr --model_weights=DEFAULT
    done
done


