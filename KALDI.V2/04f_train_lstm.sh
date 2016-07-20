. 00_init_paths.sh
local/nnet3/run_lstm.sh --affix bidirectional \
                         --lstm-delay "[-1,1] [-2,2] [-3,3]" \
                         --label-delay 5 \
                         --cell-dim 512 \
                         --num-lstm-layers 3 \
                         --hidden-dim 1024 \
                         --splice-indexes "-2,-1,0,1,2 0 0" \
                         --recurrent-projection-dim 512 \
                         --non-recurrent-projection-dim 512 \
                         --chunk-left-context 40 \
                         --chunk-right-context 40 \
                         --num-epochs 20 \
                         --initial-effective-lrate 0.0006 \
                         --final-effective-lrate 0.00006 \
                         --num-jobs-initial 2 \
                         --num-jobs-final 6