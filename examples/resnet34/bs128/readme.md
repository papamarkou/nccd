## Order of execution

1. run_lr_tuning.sh
2. run_training.sh
3. run_thres_tuning.sh
4. run_testing.sh

## Setting hyper-parameters

* After running `run_lr_tuning.sh`, set `lr_lower` and `lr_upper` in `train.sh`.
* After running `run_thres_tuning.sh`, set `thres` in `predict_test_images.sh`.
