python run.py -m RecNet -d CDs -u 6 -k 25 -tau 0.85 \
--train_batch_size=20 --eval_batch_size=200 \
--max_his_len=20 --MAX_ITEM_LIST_LENGTH=20 \
--epochs=1 --shuffle=False \
--test_only=False
