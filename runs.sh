#!/bin/sh

RUN="$1"
task="$2"
gpu_number="$3"
checkpoints_selection="$4"

work_dir=$(pwd)
MAIN_DIR=${work_dir%/*}
MAIN_DIR=${MAIN_DIR%/*}
DATA_DIR="$MAIN_DIR/Datasets/MRBrainS18"
FILES_CHECKPOINT="$work_dir/train/run_$RUN/checkpoints/run_${RUN}_data.p"
TRAIN_SUBJECTS=5
MODEL="cnn_3d_1"
LOSS_TYPE="log_loss"
HUBER_DELTA=1
PATCH_SIZE="8,24,24"
BATCH_SIZE=64
KEEP_CHECKPOINT=6
MAX_STEPS=200000
STEPS_TO_VAL=2000
LEARNING_RATE=1e-4
LEARNING_RATE_DECREASE=0.1
STEPS_TO_LEARNING_RATE_UPDATE=15000
STEPS_TO_SAVE_CHECKPOINT=1000
CHECKPOINT_PATH="$work_dir/train/run_$RUN/checkpoints/run_${RUN}"

source "$work_dir/train/run_$RUN/run_$RUN.sh"


if [ "$gpu_number" == "" ]; then
  CUDA_DEVICE="0"
else
  CUDA_DEVICE="$gpu_number"
fi
echo "Selected GPU: $CUDA_DEVICE"


if [ "$checkpoints_selection" == "" ]; then
  TEST_CHECKPOINTS="$MAX_STEPS"
else
  TEST_CHECKPOINTS="$checkpoints_selection"
fi
echo "Checkpoint: $TEST_CHECKPOINTS"


if [ "$task" == "train" ]; then

  mkdir "train/run_${RUN}/checkpoints"

  python mrbrains18_train.py \
    --data_dir="$DATA_DIR" \
    --files_checkpoint="$FILES_CHECKPOINT" \
    --train_subjects=$TRAIN_SUBJECTS \
    --model="$MODEL" \
    --loss_type="$LOSS_TYPE" \
    --huber_delta="$HUBER_DELTA" \
    --patch_size="$PATCH_SIZE" \
    --batch_size=$BATCH_SIZE \
    --keep_checkpoint=$KEEP_CHECKPOINT \
    --max_steps=$MAX_STEPS \
    --steps_to_val=$STEPS_TO_VAL \
    --learning_rate=$LEARNING_RATE \
    --learning_rate_decrease=$LEARNING_RATE_DECREASE \
    --steps_to_learning_rate_update=$STEPS_TO_LEARNING_RATE_UPDATE \
    --steps_to_save_checkpoint=$STEPS_TO_SAVE_CHECKPOINT \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --test_checkpoints="$TEST_CHECKPOINTS" \
    --cuda_device="$CUDA_DEVICE" 


elif [ "$task" == "test" ]; then

  python mrbrains18_test.py \
    --data_dir="$DATA_DIR" \
    --files_checkpoint="$FILES_CHECKPOINT" \
    --train_subjects=$TRAIN_SUBJECTS \
    --model="$MODEL" \
    --loss_type="$LOSS_TYPE" \
    --huber_delta="$HUBER_DELTA" \
    --patch_size="$PATCH_SIZE" \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --test_checkpoints="$TEST_CHECKPOINTS" \
    --cuda_device="$CUDA_DEVICE" 


elif [ "$task" == "summaries" ]; then

  FILES=$(ls "train/run_${RUN}/checkpoints" | grep ".index")

  TEST_CHECKPOINTS=""

  for file in $FILES
  do

    CHECKPOINT=${file%.*}
    TEST_CHECKPOINTS="$TEST_CHECKPOINTS,${CHECKPOINT##*-}"

  done

  TEST_CHECKPOINTS=${TEST_CHECKPOINTS#*,}
  TEST_CHECKPOINTS=$(echo $TEST_CHECKPOINTS | tr "," "\n" | sort -h | tr "\n" ",")
  TEST_CHECKPOINTS=${TEST_CHECKPOINTS%,*}
  echo $TEST_CHECKPOINTS

  mkdir "train/run_${RUN}/summaries"

  python mrbrains18_test.py \
    --data_dir="$DATA_DIR" \
    --files_checkpoint="$FILES_CHECKPOINT" \
    --train_subjects=$TRAIN_SUBJECTS \
    --model="$MODEL" \
    --loss_type="$LOSS_TYPE" \
    --huber_delta="$HUBER_DELTA" \
    --patch_size="$PATCH_SIZE" \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --test_checkpoints="$TEST_CHECKPOINTS" \
    --cuda_device="$CUDA_DEVICE" \
    > "train/run_${RUN}/summaries/run_${RUN}_summary.txt"


else

  echo "*****************************************************************"
  echo "Put the arguments in the following order:"
  echo "1. Run number: Integer value"
  echo "2. Task: String between train, test or summaries"
  echo "3. GPU Device Number: Integer value"
  echo "4. Checkpoints to test: Checkpoint number"
  echo "*****************************************************************"

fi