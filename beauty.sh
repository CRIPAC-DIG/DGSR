#!/user/bin/env/ bash
 nohup  python -u  new_main.py \
 --data=Beauty \
 --gpu=0 \
 --epoch=20 \
 --user_long=orgat \
 --user_short=att \
 --item_long=orgat \
 --item_short=att \
 --user_update=rnn \
 --item_update=rnn \
 --batchSize=50 \
 --layer_num=3 \
 --lr=0.001 \
 --l2=0.00001 \
 --item_max_length=50 \
 --user_max_length=50 \
 --record \
 --model_record \
 >./jup&







