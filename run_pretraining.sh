nohup python run_pretraining.py \
--bert_config_file pretrained/medbert-wwm-base/config.json \
--input_file data/mdg/train.tfrecord \
--output_dir checkpoint/medbert-wwm-ft \
--init_checkpoint pretrained/medbert-wwm-base/medbert_wwm_base.ckpt.data-00000-of-00001 \
--max_seq_length 256 \
--max_predictions_per_seq 20 \
--do_train True \
--train_batch_size 32 \
>> checkpoint/fine-tune.log 2>&1 &