#!/user/bin/env/ bash
nohup  python -u  new_data.py \
 --data=Beauty \
 --job=10 \
 --item_max_length=50 \
 --user_max_length=50 \
 --k_hop=3 \
 >./results/be_data&

