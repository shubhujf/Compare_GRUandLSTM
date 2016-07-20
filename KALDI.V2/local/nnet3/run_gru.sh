#!/bin/bash

# this is a basic Gated Recurrent Unit(GRU) script. GRU is a kind of recurrent
# neural network similar to LSTM, but faster and less likely to diverge than LSTM.
# See http://arxiv.org/pdf/1512.02595v1.pdf for more info about the network.
# GRU script runs for more epochs than the TDNN script
# and each epoch takes twice the time

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call gru/train.sh with --gpu false

stage=0
train_stage=-10
affix=
common_egs_dir=

# GRU options
splice_indexes="-2,-1,0,1,2 0 0"
gru_delay=" -1 -2 -3 -4 -5 "
label_delay=5
num_gru_layers=5
recurrent_projection_dim=1024
non_recurrent_projection_dim=512
hidden_dim=1024
chunk_width=20
chunk_left_context=40
chunk_right_context=0


# training options
num_epochs=15
initial_effective_lrate=0.0006
final_effective_lrate=0.00006
num_jobs_initial=2
num_jobs_final=12
momentum=0.5
num_chunk_per_minibatch=100
samples_per_iter=20000
remove_egs=true

#decode options
extra_left_context=
extra_right_context=
frames_per_chunk=

#End configuration section

echo "$0 $@" # Print the command line for logging

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

dir=exp/nnet3/bgru_l3
dir=$dir${affix:+_$affix}
if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi

#local/nnet3/run_ivector_common.sh --stage $stage || exit 1;
# --online-ivector-dir exp/nnet3/ivectors_train_si284 \

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/gru/train.sh  \
    --label-delay $label_delay \
    --gru-delay "$gru_delay" \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --num-chunk-per-minibatch $num_chunk_per_minibatch \
    --samples-per-iter $samples_per_iter \
    --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --momentum $momentum \
    --cmd "$decode_cmd" \
    --num-gru-layers $num_gru_layers \
    --recurrent-projection-dim $recurrent_projection_dim \
    --non-recurrent-projection-dim $non_recurrent_projection_dim \
    --hidden-dim $hidden_dim \
    --chunk-width $chunk_width \
    --chunk-left-context $chunk_left_context \
    --chunk-right-context $chunk_right_context \
    --egs-dir "$common_egs_dir" \
    --remove-egs $remove_egs \
     data/train lang_train exp/tri3b_ali $dir  || exit 1;
fi

#if [ $stage -le 9 ]; then
#  if [ -z $extra_left_context ]; then
#    extra_left_context=$chunk_left_context
#  fi
#  if [ -z $extra_right_context ]; then
#    extra_right_context=$chunk_right_context
#  fi
#  if [ -z $frames_per_chunk ]; then
#    frames_per_chunk=$chunk_width
#  fi
#  for lm_suffix in tgpr bd_tgpr; do
#    graph_dir=exp/tri4b/graph_${lm_suffix}
#    # use already-built graphs
#    for year in eval92 dev93; do
#      (
#      num_jobs=`cat data/test_${year}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
#      steps/nnet3/lstm/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
#        --extra-left-context $extra_left_context \
#        --extra-right-context $extra_right_context \
#        --frames-per-chunk "$frames_per_chunk" \
#        --online-ivector-dir exp/nnet3/ivectors_test_$year \
#        $graph_dir data/test_${year}_hires $dir/decode_${lm_suffix}_${year} || exit 1;
#      ) &
#    done
#  done
#fi

if [ $stage -le 9 ]; then
  if [ -z $extra_left_context ]; then
    extra_left_context=$chunk_left_context
  fi  
  if [ -z $extra_right_context ]; then
    extra_right_context=$chunk_right_context
  fi  
  if [ -z $frames_per_chunk ]; then
    frames_per_chunk=$chunk_width
  fi  
  steps/nnet3/lstm/decode.sh --nj 8 --cmd "$decode_cmd" --extra-left-context $extra_left_context --extra-right-context $extra_right_context --frames-per-chunk "$frames_per_chunk" exp/tri3b/graph data/dev exp/nnet3/bgru_l3_bidirectional_ld5/decode_dev
fi

exit 0;


