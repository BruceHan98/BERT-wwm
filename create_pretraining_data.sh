nohup python create_pretraining_data.py \
--input_file data/mdg/train_dialog_seg.txt \
--output_file data/mdg/train.tfrecord \
--vocab_file pretrained/medbert-wwm-base/vocab.txt \
--do_lower_case True \
--do_whole_word_mask True \
--max_seq_length 256 \
--max_predictions_per_seq 20 \
--dupe_factor 5 \
--masked_lm_prob 0.15 \
>> create_train_data.log 2>&1 &