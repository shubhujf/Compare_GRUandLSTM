#!/usr/bin/env python

# Gated Recurrent Unit(GRU) is a kind of recurrent neural network similar to LSTM, but faster and less likely to diverge than LSTM.
# See http://arxiv.org/pdf/1512.02595v1.pdf for more info about the network.

from __future__ import print_function
import os
import argparse
import sys
import warnings
import copy
import imp

nodes = imp.load_source('', 'steps/nnet3/components.py')

def PrintConfig(file_name, config_lines):
    f = open(file_name, 'w')
    f.write("\n".join(config_lines['components'])+"\n")
    f.write("\n#Component nodes\n")
    f.write("\n".join(config_lines['component-nodes'])+"\n")
    f.close()

def WriteScaleMinusOne(file_name, recurrent_projection_dim):
    f = open(file_name, 'w')
    f.write(" [ ")
    for i in range(recurrent_projection_dim):
        f.write("-1 ")
    f.write("]\n")
    f.close()

def WriteBiasOne(file_name, recurrent_projection_dim):
    f = open(file_name, 'w')
    f.write(" [ ")
    for i in range(recurrent_projection_dim):
        f.write("1 ")
    f.write("]\n")
    f.close()

def ParseSpliceString(splice_indexes, label_delay=None):
    ## Work out splice_array e.g. splice_array = [ [ -3,-2,...3 ], [0], [-2,2], .. [ -8,8 ] ]
    split1 = splice_indexes.split(" ");  # we already checked the string is nonempty.
    if len(split1) < 1:
        splice_indexes = "0"

    left_context=0
    right_context=0
    if label_delay is not None:
        left_context = -label_delay
        right_context = label_delay

    splice_array = []
    try:
        for i in range(len(split1)):
            indexes = map(lambda x: int(x), split1[i].strip().split(","))
            print(indexes)
            if len(indexes) < 1:
                raise ValueError("invalid --splice-indexes argument, too-short element: "
                                + splice_indexes)

            if (i > 0)  and ((len(indexes) != 1) or (indexes[0] != 0)):
                raise ValueError("elements of --splice-indexes splicing is only allowed initial layer.")

            if not indexes == sorted(indexes):
                raise ValueError("elements of --splice-indexes must be sorted: "
                                + splice_indexes)
            left_context += -indexes[0]
            right_context += indexes[-1]
            splice_array.append(indexes)
    except ValueError as e:
        raise ValueError("invalid --splice-indexes argument " + splice_indexes + str(e))

    left_context = max(0, left_context)
    right_context = max(0, right_context)

    return {'left_context':left_context,
            'right_context':right_context,
            'splice_indexes':splice_array,
            'num_hidden_layers':len(splice_array)
            }

def ParseGruDelayString(gru_delay):
    ## Work out gru_delay e.g. "-1 [-1,1] -2" -> list([ [-1], [-1, 1], [-2] ])
    split1 = gru_delay.split(" ");
    gru_delay_array = []
    try:
        for i in range(len(split1)):
            indexes = map(lambda x: int(x), split1[i].strip().lstrip('[').rstrip(']').strip().split(","))
            if len(indexes) < 1:
                raise ValueError("invalid --gru-delay argument, too-short element: "
                                + gru_delay)
            elif len(indexes) == 2 and indexes[0] * indexes[1] >= 0:
                raise ValueError('Warning: ' + str(indexes) + ' is not a standard bidirectional mode. There should be a negative delay for the forward, and a postive delay for the backward.')
            gru_delay_array.append(indexes)
    except ValueError as e:
        raise ValueError("invalid --gru-delay argument " + gru_delay + str(e))

    return gru_delay_array

   
if __name__ == "__main__":
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Writes config files and variables "
                                                 "for GRUs creation and training",
                                     epilog="See steps/nnet3/gru/train.sh for example.")
    # General neural network options
    parser.add_argument("--splice-indexes", type=str,
                        help="Splice indexes at input layer, e.g. '-3,-2,-1,0,1,2,3' [compulsary argument]", default="0")
    parser.add_argument("--feat-dim", type=int,
                        help="Raw feature dimension, e.g. 13")
    parser.add_argument("--ivector-dim", type=int,
                        help="iVector dimension, e.g. 100", default=0)
    parser.add_argument("--include-log-softmax", type=str,
                        help="add the final softmax layer ", default="true", choices = ["false", "true"])

    # GRU options
    parser.add_argument("--num-gru-layers", type=int,
                        help="Number of GRU layers to be stacked", default=1)
    parser.add_argument("--recurrent-projection-dim", type=int,
                        help="dimension of gru's recurrent projection node")
    parser.add_argument("--non-recurrent-projection-dim", type=int,
                        help="dimension of gru's non-recurrent projection node") 
    parser.add_argument("--hidden-dim", type=int,
                        help="dimension of fully-connected layers")

    # Natural gradient options
    parser.add_argument("--ng-affine-options", type=str,
                        help="options to be supplied to NaturalGradientAffineComponent", default="")

    # Gradient clipper options
    parser.add_argument("--norm-based-clipping", type=str,
                        help="use norm based clipping in ClipGradient components ", default="false", choices = ["false", "true"])
    parser.add_argument("--clipping-threshold", type=float,
                        help="clipping threshold used in ClipGradient components, if clipping-threshold=0 no clipping is done", default=15)

    parser.add_argument("--num-targets", type=int,
                        help="number of network targets (e.g. num-pdf-ids/num-leaves)")
    parser.add_argument("config_dir",
                        help="Directory to write config files and variables")

    # Delay options
    parser.add_argument("--label-delay", type=int, default=None,
                        help="option to delay the labels to make the lstm robust")

    parser.add_argument("--gru-delay", type=str, default=None,
                        help="option to have different delays in recurrence for each lstm")



    print(' '.join(sys.argv))

    args = parser.parse_args()

    if not os.path.exists(args.config_dir):
        os.makedirs(args.config_dir)

    ## Check arguments.
    if args.splice_indexes is None:
        sys.exit("--splice-indexes argument is required")
    if args.feat_dim is None or not (args.feat_dim > 0):
        sys.exit("--feat-dim argument is required")
    if args.num_targets is None or not (args.num_targets > 0):
        sys.exit("--num-targets argument is required")
    if (args.num_gru_layers < 1):
        sys.exit("--num-gru-layers has to be a positive integer")
    if (args.clipping_threshold < 0):
        sys.exit("--clipping-threshold has to be a non-negative")
    if args.gru_delay is None:
        gru_delay = [-1] * args.num_gru_layers
    else:
        try:
            gru_delay = ParseGruDelayString(args.gru_delay.strip())
        except ValueError:
            sys.exit("--gru-delay has incorrect format value. Provided value is '{0}'".format(args.gru_delay))
        if len(gru_delay) != args.num_gru_layers:
            sys.exit("--gru-delay: Number of delays provided has to match --num-gru-layers")

    parsed_splice_output = ParseSpliceString(args.splice_indexes.strip(), args.label_delay)
    left_context = parsed_splice_output['left_context']
    right_context = parsed_splice_output['right_context']
    num_hidden_layers = parsed_splice_output['num_hidden_layers']
    splice_indexes = parsed_splice_output['splice_indexes']

    #if (num_hidden_layers < args.num_gru_layers):
    #    sys.exit("--num-gru-layers : number of gru layers has to be no greater than number of hidden layers, decided based on splice-indexes")

    # write the files used by other scripts like steps/nnet3/get_egs.sh
    f = open(args.config_dir + "/vars", "w")
    print('model_left_context=' + str(left_context), file=f)
    print('model_right_context=' + str(right_context), file=f)
    print('num_hidden_layers=' + str(num_hidden_layers), file=f)
    # print('initial_right_context=' + str(splice_array[0][-1]), file=f)
    f.close()

    WriteScaleMinusOne(args.config_dir + '/scale_minus_one.vec', args.recurrent_projection_dim)
    WriteBiasOne(args.config_dir + '/bias_one.vec', args.recurrent_projection_dim)

    config_lines = {'components':[], 'component-nodes':[]}

    config_files={}
    prev_layer_output = nodes.AddInputLayer(config_lines, args.feat_dim, splice_indexes[0], args.ivector_dim)

    # Add the init config lines for estimating the preconditioning matrices
    init_config_lines = copy.deepcopy(config_lines)
    init_config_lines['components'].insert(0, '# Config file for initializing neural network prior to')
    init_config_lines['components'].insert(0, '# preconditioning matrix computation')
    nodes.AddOutputLayer(init_config_lines, prev_layer_output)
    config_files[args.config_dir + '/init.config'] = init_config_lines

    prev_layer_output = nodes.AddLdaLayer(config_lines, "L0", prev_layer_output, args.config_dir + '/lda.mat')
    print('Value of NUM GRU LAYERS is '+ str(args.num_gru_layers))

    #for i in range(0, 2):
    #    prev_layer_output = nodes.AddAffRelNormLayer(config_lines, "DNN_before_GRU{0}".format(i+1), prev_layer_output, args.hidden_dim, args.ng_affine_options)


    for i in range(args.num_gru_layers):
        if len(gru_delay[i]) == 2: # Birectional Gru layer case, add both forward and backward
            prev_layer_output1 = nodes.AddGruLayer(config_lines, "BGru{0}_forward".format(i+1), prev_layer_output, args.recurrent_projection_dim, args.non_recurrent_projection_dim,
                                            args.config_dir + '/scale_minus_one.vec', args.config_dir + '/bias_one.vec',
                                            args.clipping_threshold, args.norm_based_clipping, args.ng_affine_options, gru_delay = gru_delay[i][0])
            prev_layer_output2 = nodes.AddGruLayer(config_lines, "BGru{0}_backward".format(i+1), prev_layer_output, args.recurrent_projection_dim, args.non_recurrent_projection_dim,
                                            args.config_dir + '/scale_minus_one.vec', args.config_dir + '/bias_one.vec',
                                            args.clipping_threshold, args.norm_based_clipping, args.ng_affine_options, gru_delay = gru_delay[i][1])
            prev_layer_output['descriptor'] = 'Append({0}, {1})'.format(prev_layer_output1['descriptor'], prev_layer_output2['descriptor'])
            prev_layer_output['dimension'] = prev_layer_output1['dimension'] + prev_layer_output2['dimension']
        else: # Gru layer case
            prev_layer_output = nodes.AddGruLayer(config_lines, "Gru{0}".format(i+1), prev_layer_output, args.recurrent_projection_dim, args.non_recurrent_projection_dim,
			                                args.config_dir + '/scale_minus_one.vec', args.config_dir + '/bias_one.vec',
                                            args.clipping_threshold, args.norm_based_clipping, args.ng_affine_options, gru_delay = gru_delay[i][0])
        # make the intermediate config file for layerwise discriminative training
        nodes.AddFinalLayer(config_lines, prev_layer_output, args.num_targets, args.ng_affine_options, args.label_delay, include_log_softmax = args.include_log_softmax)
        config_files['{0}/layer{1}.config'.format(args.config_dir, i+1)] = config_lines
        config_lines = {'components':[], 'component-nodes':[]}
    print('Value of NUM HIDDEN LAYERS is '+ str(num_hidden_layers))
    print('Value of NUM GRU LAYERS is '+ str(args.num_gru_layers,))
    for i in range(args.num_gru_layers, num_hidden_layers):
        prev_layer_output = nodes.AddAffRelNormLayer(config_lines, "DNN_after_GRU{0}".format(i+1), prev_layer_output, args.hidden_dim, args.ng_affine_options)
        # make the intermediate config file for layerwise discriminative training
        nodes.AddFinalLayer(config_lines, prev_layer_output, args.num_targets, args.ng_affine_options, args.label_delay, include_log_softmax = args.include_log_softmax)
        config_files['{0}/layer{1}.config'.format(args.config_dir, i+1)] = config_lines
        config_lines = {'components':[], 'component-nodes':[]}

    # printing out the configs
    # init.config used to train lda-mllt train
    for key in config_files.keys():
        PrintConfig(key, config_files[key])

