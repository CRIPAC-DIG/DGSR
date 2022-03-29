#!/user/bin/env/ bash
nohup  python -u  new_data.py \
 --data=Games \
 --job=10 \
 --item_max_length=50 \
 --user_max_length=50 \
 --k_hop=2 \
 >./results/ga_data&

