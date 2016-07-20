#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import sys
import warnings
import copy
from operator import itemgetter

def GetSumDescriptor(inputs):
    sum_descriptors = inputs
    while len(sum_descriptors) != 1:
        cur_sum_descriptors = []
        pair = []
        while len(sum_descriptors) > 0:
            value = sum_descriptors.pop()
            if value.strip() != '':
                pair.append(value)
            if len(pair) == 2:
                cur_sum_descriptors.append("Sum({0}, {1})".format(pair[0], pair[1]))
                pair = []
        if pair:
            cur_sum_descriptors.append(pair[0])
        sum_descriptors = cur_sum_descriptors
    return sum_descriptors

# adds the input nodes and returns the descriptor
def AddInputLayer(config_lines, feat_dim, splice_indexes=[0], ivector_dim=0):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    output_dim = 0
    components.append('input-node name=input dim=' + str(feat_dim))
    list = [('Offset(input, {0})'.format(n) if n != 0 else 'input') for n in splice_indexes]
    output_dim += len(splice_indexes) * feat_dim
    if ivector_dim > 0:
        components.append('input-node name=ivector dim=' + str(ivector_dim))
        list.append('ReplaceIndex(ivector, t, 0)')
        output_dim += ivector_dim
    if len(list) > 1:
        splice_descriptor = "Append({0})".format(", ".join(list))
    else:
        splice_descriptor = list[0]
    print(splice_descriptor)
    return {'descriptor': splice_descriptor,
            'dimension': output_dim}

def AddNoOpLayer(config_lines, name, input):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append('component name={0}_noop type=NoOpComponent dim={1}'.format(name, input['dimension']))
    component_nodes.append('component-node name={0}_noop component={0}_noop input={1}'.format(name, input['descriptor']))

    return {'descriptor':  '{0}_noop'.format(name),
            'dimension': input['dimension']}

def AddLdaLayer(config_lines, name, input, lda_file):
    return AddFixedAffineLayer(config_lines, name, input, lda_file)

def AddFixedAffineLayer(config_lines, name, input, matrix_file):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append('component name={0}_fixaffine type=FixedAffineComponent matrix={1}'.format(name, matrix_file))
    component_nodes.append('component-node name={0}_fixaffine component={0}_fixaffine input={1}'.format(name, input['descriptor']))

    return {'descriptor':  '{0}_fixaffine'.format(name),
            'dimension': input['dimension']}


def AddBlockAffineLayer(config_lines, name, input, output_dim, num_blocks):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    assert((input['dimension'] % num_blocks == 0) and
            (output_dim % num_blocks == 0))
    components.append('component name={0}_block_affine type=BlockAffineComponent input-dim={1} output-dim={2} num-blocks={3}'.format(name, input['dimension'], output_dim, num_blocks))
    component_nodes.append('component-node name={0}_block_affine component={0}_block_affine input={1}'.format(name, input['descriptor']))

    return {'descriptor' : '{0}_block_affine'.format(name),
                           'dimension' : output_dim}

def AddPermuteLayer(config_lines, name, input, column_map):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    permute_indexes = ",".join(map(lambda x: str(x), column_map))
    components.append('component name={0}_permute type=PermuteComponent column-map={1}'.format(name, permute_indexes))
    component_nodes.append('component-node name={0}_permute component={0}_permute input={1}'.format(name, input['descriptor']))

    return {'descriptor': '{0}_permute'.format(name),
            'dimension': input['dimension']}

def AddAffineLayer(config_lines, name, input, output_dim, ng_affine_options = ""):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))

    return {'descriptor':  '{0}_affine'.format(name),
            'dimension': output_dim}

def AddAffRelNormLayer(config_lines, name, input, output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0, self_repair_scale = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, self_repair_string))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, output_dim, norm_target_rms))

    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={0}_affine".format(name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu".format(name))

    return {'descriptor':  '{0}_renorm'.format(name),
            'dimension': output_dim}

def AddConvolutionLayer(config_lines, name, input,
                       input_x_dim, input_y_dim, input_z_dim,
                       filt_x_dim, filt_y_dim,
                       filt_x_step, filt_y_step,
                       num_filters, input_vectorization,
                       param_stddev = None, bias_stddev = None,
                       filter_bias_file = None,
                       is_updatable = True):
    assert(input['dimension'] == input_x_dim * input_y_dim * input_z_dim)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    conv_init_string = ("component name={name}_conv type=ConvolutionComponent "
                       "input-x-dim={input_x_dim} input-y-dim={input_y_dim} input-z-dim={input_z_dim} "
                       "filt-x-dim={filt_x_dim} filt-y-dim={filt_y_dim} "
                       "filt-x-step={filt_x_step} filt-y-step={filt_y_step} "
                       "input-vectorization-order={vector_order}".format(name = name,
                       input_x_dim = input_x_dim, input_y_dim = input_y_dim, input_z_dim = input_z_dim,
                       filt_x_dim = filt_x_dim, filt_y_dim = filt_y_dim,
                       filt_x_step = filt_x_step, filt_y_step = filt_y_step,
                       vector_order = input_vectorization))
    if filter_bias_file is not None:
        conv_init_string += " matrix={0}".format(filter_bias_file)
    else:
        conv_init_string += " num-filters={0}".format(num_filters)

    components.append(conv_init_string)
    component_nodes.append("component-node name={0}_conv_t component={0}_conv input={1}".format(name, input['descriptor']))

    num_x_steps = (1 + (input_x_dim - filt_x_dim) / filt_x_step)
    num_y_steps = (1 + (input_y_dim - filt_y_dim) / filt_y_step)
    output_dim = num_x_steps * num_y_steps * num_filters;
    return {'descriptor':  '{0}_conv_t'.format(name),
            'dimension': output_dim,
            '3d-dim': [num_x_steps, num_y_steps, num_filters],
            'vectorization': 'zyx'}

# The Maxpooling component assumes input vectorizations of type zyx
def AddMaxpoolingLayer(config_lines, name, input,
                      input_x_dim, input_y_dim, input_z_dim,
                      pool_x_size, pool_y_size, pool_z_size,
                      pool_x_step, pool_y_step, pool_z_step):
    if input_x_dim < 1 or input_y_dim < 1 or input_z_dim < 1:
        raise Exception("non-positive maxpooling input size ({0}, {1}, {2})".
                 format(input_x_dim, input_y_dim, input_z_dim))
    if pool_x_size > input_x_dim or pool_y_size > input_y_dim or pool_z_size > input_z_dim:
        raise Exception("invalid maxpooling pool size vs. input size")
    if pool_x_step > pool_x_size or pool_y_step > pool_y_size or pool_z_step > pool_z_size:
        raise Exception("invalid maxpooling pool step vs. pool size")
    
    assert(input['dimension'] == input_x_dim * input_y_dim * input_z_dim)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append('component name={name}_maxp type=MaxpoolingComponent '
                      'input-x-dim={input_x_dim} input-y-dim={input_y_dim} input-z-dim={input_z_dim} '
                      'pool-x-size={pool_x_size} pool-y-size={pool_y_size} pool-z-size={pool_z_size} '
                      'pool-x-step={pool_x_step} pool-y-step={pool_y_step} pool-z-step={pool_z_step} '.
                      format(name = name,
                      input_x_dim = input_x_dim, input_y_dim = input_y_dim, input_z_dim = input_z_dim,
                      pool_x_size = pool_x_size, pool_y_size = pool_y_size, pool_z_size = pool_z_size,
                      pool_x_step = pool_x_step, pool_y_step = pool_y_step, pool_z_step = pool_z_step))

    component_nodes.append('component-node name={0}_maxp_t component={0}_maxp input={1}'.format(name, input['descriptor']))

    num_pools_x = 1 + (input_x_dim - pool_x_size) / pool_x_step;
    num_pools_y = 1 + (input_y_dim - pool_y_size) / pool_y_step;
    num_pools_z = 1 + (input_z_dim - pool_z_size) / pool_z_step;
    output_dim = num_pools_x * num_pools_y * num_pools_z;

    return {'descriptor':  '{0}_maxp_t'.format(name),
            'dimension': output_dim,
            '3d-dim': [num_pools_x, num_pools_y, num_pools_z],
            'vectorization': 'zyx'}


def AddSoftmaxLayer(config_lines, name, input):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_log_softmax type=LogSoftmaxComponent dim={1}".format(name, input['dimension']))
    component_nodes.append("component-node name={0}_log_softmax component={0}_log_softmax input={1}".format(name, input['descriptor']))

    return {'descriptor':  '{0}_log_softmax'.format(name),
            'dimension': input['dimension']}


def AddSigmoidLayer(config_lines, name, input, self_repair_scale = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_sigmoid type=SigmoidComponent dim={1}".format(name, input['dimension'], self_repair_string))
    component_nodes.append("component-node name={0}_sigmoid component={0}_sigmoid input={1}".format(name, input['descriptor']))
    return {'descriptor':  '{0}_sigmoid'.format(name),
            'dimension': input['dimension']}

def AddOutputLayer(config_lines, input, label_delay = None, suffix=None, objective_type = "linear"):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    name = 'output'
    if suffix is not None:
        name = '{0}-{1}'.format(name, suffix)

    if label_delay is None:
        component_nodes.append('output-node name={0} input={1} objective={2}'.format(name, input['descriptor'], objective_type))
    else:
        component_nodes.append('output-node name={0} input=Offset({1},{2}) objective={3}'.format(name, input['descriptor'], label_delay, objective_type))

def AddFinalLayer(config_lines, input, output_dim,
        ng_affine_options = " param-stddev=0 bias-stddev=0 ",
        label_delay=None,
        use_presoftmax_prior_scale = False,
        prior_scale_file = None,
        include_log_softmax = True,
        add_final_sigmoid = False,
        name_affix = None,
        objective_type = "linear"):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    if name_affix is not None:
        final_node_prefix = 'Final-' + str(name_affix)
    else:
        final_node_prefix = 'Final'

    prev_layer_output = AddAffineLayer(config_lines,
            final_node_prefix , input, output_dim,
            ng_affine_options)
    if include_log_softmax:
        if use_presoftmax_prior_scale :
            components.append('component name={0}-fixed-scale type=FixedScaleComponent scales={1}'.format(final_node_prefix, prior_scale_file))
            component_nodes.append('component-node name={0}-fixed-scale component={0}-fixed-scale input={1}'.format(final_node_prefix,
                prev_layer_output['descriptor']))
            prev_layer_output['descriptor'] = "{0}-fixed-scale".format(final_node_prefix)
        prev_layer_output = AddSoftmaxLayer(config_lines, final_node_prefix, prev_layer_output)
    elif add_final_sigmoid:
        # Useful when you need the final outputs to be probabilities
        # between 0 and 1.
        # Usually used with an objective-type such as "quadratic"
        prev_layer_output = AddSigmoidLayer(config_lines, final_node_prefix, prev_layer_output)
    # we use the same name_affix as a prefix in for affine/scale nodes but as a
    # suffix for output node
    AddOutputLayer(config_lines, prev_layer_output, label_delay, suffix = name_affix, objective_type = objective_type)

def AddLstmLayer(config_lines,
                 name, input, cell_dim,
                 recurrent_projection_dim = 0,
                 non_recurrent_projection_dim = 0,
                 clipping_threshold = 1.0,
                 norm_based_clipping = "false",
                 ng_per_element_scale_options = "",
                 ng_affine_options = "",
                 lstm_delay = -1,
                 self_repair_scale = None):
    assert(recurrent_projection_dim >= 0 and non_recurrent_projection_dim >= 0)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    input_descriptor = input['descriptor']
    input_dim = input['dimension']
    name = name.strip()

    if (recurrent_projection_dim == 0):
        add_recurrent_projection = False
        recurrent_projection_dim = cell_dim
        recurrent_connection = "m_t"
    else:
        add_recurrent_projection = True
        recurrent_connection = "r_t"
    if (non_recurrent_projection_dim == 0):
        add_non_recurrent_projection = False
    else:
        add_non_recurrent_projection = True

    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    # Natural gradient per element scale parameters
    ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
    # Parameter Definitions W*(* replaced by - to have valid names)
    components.append("# Input gate control : W_i* matrices")
    components.append("component name={0}_W_i-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_ic type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("# Forget gate control : W_f* matrices")
    components.append("component name={0}_W_f-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_fc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("#  Output gate control : W_o* matrices")
    components.append("component name={0}_W_o-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_oc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("# Cell input matrices : W_c* matrices")
    components.append("component name={0}_W_c-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))


    components.append("# Defining the non-linearities")
    components.append("component name={0}_i type=SigmoidComponent dim={1} {2}".format(name, cell_dim, self_repair_string))
    components.append("component name={0}_f type=SigmoidComponent dim={1} {2}".format(name, cell_dim, self_repair_string))
    components.append("component name={0}_o type=SigmoidComponent dim={1} {2}".format(name, cell_dim, self_repair_string))
    components.append("component name={0}_g type=TanhComponent dim={1} {2}".format(name, cell_dim, self_repair_string))
    components.append("component name={0}_h type=TanhComponent dim={1} {2}".format(name, cell_dim, self_repair_string))

    components.append("# Defining the cell computations")
    components.append("component name={0}_c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, cell_dim, clipping_threshold, norm_based_clipping))

    # c1_t and c2_t defined below
    component_nodes.append("component-node name={0}_c_t component={0}_c input=Sum({0}_c1_t, {0}_c2_t)".format(name))
    c_tminus1_descriptor = "IfDefined(Offset({0}_c_t, {1}))".format(name, lstm_delay)

    component_nodes.append("# i_t")
    component_nodes.append("component-node name={0}_i1 component={0}_W_i-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_i2 component={0}_w_ic  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_i_t component={0}_i input=Sum({0}_i1, {0}_i2)".format(name))

    component_nodes.append("# f_t")
    component_nodes.append("component-node name={0}_f1 component={0}_W_f-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_f2 component={0}_w_fc  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_f_t component={0}_f input=Sum({0}_f1,{0}_f2)".format(name))

    component_nodes.append("# o_t")
    component_nodes.append("component-node name={0}_o1 component={0}_W_o-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_o2 component={0}_w_oc input={0}_c_t".format(name))
    component_nodes.append("component-node name={0}_o_t component={0}_o input=Sum({0}_o1, {0}_o2)".format(name))

    component_nodes.append("# h_t")
    component_nodes.append("component-node name={0}_h_t component={0}_h input={0}_c_t".format(name))

    component_nodes.append("# g_t")
    component_nodes.append("component-node name={0}_g1 component={0}_W_c-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_g_t component={0}_g input={0}_g1".format(name))

    component_nodes.append("# parts of c_t")
    component_nodes.append("component-node name={0}_c1_t component={0}_c1  input=Append({0}_f_t, {1})".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_c2_t component={0}_c2 input=Append({0}_i_t, {0}_g_t)".format(name))

    component_nodes.append("# m_t")
    component_nodes.append("component-node name={0}_m_t component={0}_m input=Append({0}_o_t, {0}_h_t)".format(name))

    # add the recurrent connections
    if (add_recurrent_projection and add_non_recurrent_projection):
        components.append("# projection matrices : Wrm and Wpm")
        components.append("component name={0}_W-m type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim + non_recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("# r_t and p_t")
        component_nodes.append("component-node name={0}_rp_t component={0}_W-m input={0}_m_t".format(name))
        component_nodes.append("dim-range-node name={0}_r_t_preclip input-node={0}_rp_t dim-offset=0 dim={1}".format(name, recurrent_projection_dim))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_rp_t'.format(name)
        output_dim = recurrent_projection_dim + non_recurrent_projection_dim

    elif add_recurrent_projection:
        components.append("# projection matrices : Wrm")
        components.append("component name={0}_Wrm type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("# r_t")
        component_nodes.append("component-node name={0}_r_t_preclip component={0}_Wrm input={0}_m_t".format(name))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = recurrent_projection_dim

    else:
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, cell_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_m_t".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = cell_dim

    return {
            'descriptor': output_descriptor,
            'dimension':output_dim
            }

# Equations that specify a Gated Recurrent Unit (GRU) layer
# (according to Eqn. (10) in the paper http://arxiv.org/pdf/1512.02595v1.pdf,
# except that we scale h(t-1) by r(t) prior to the appplication of Whh as standard GRUs):
# input: x(t), output: y(t), recurrent connection: h(t)
# r(t) = sigmoid(Wrx * x(t) + Wrh * h(t-1) + br)
# z(t) = sigmoid(Wzx * x(t) + Wzh * h(t-1) + bz)
# h_tilt(t) = tanh(Whx * x(t) + Whh * (r(t) .* h(t-1)) + bh)
# h(t) = z(t) .* h_tilt(t) + (1 - z(t)) .* h(t-1)
# p(t) = Wph * h(t) + bp
# y(t) = norm(relu(p(t)))
# where Wrh, Wzh, Whh is low-rank
def AddGruLayer(config_lines,
                name, input, recurrent_projection_dim,
                non_recurrent_projection_dim,
                scale_minus_one_file, bias_one_file,
                clipping_threshold = 1.0,
                norm_based_clipping = "false",
                ng_affine_options = "",
                gru_delay = -1):
    assert(recurrent_projection_dim > 0 and non_recurrent_projection_dim > 0)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    input_descriptor = input['descriptor']
    input_dim = input['dimension']
    name = name.strip()

    # Parameter Definitions of affine matrix W_*-xh for reset gate r(t) and update gate z(t)
    components.append("# Define affine matrices for reset gate and update gate: W_*-xh")
    components.append("component name={0}_W_r-xh type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim / 4, recurrent_projection_dim, ng_affine_options))
    components.append("component name={0}_W_z-xh type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim / 4, recurrent_projection_dim, ng_affine_options))

    # Parameter Definitions W_h-xh
    components.append("# Define the affine matrix for current hidden output control: W_h-xh")
    components.append("component name={0}_W_h-xh type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim / 4, recurrent_projection_dim, ng_affine_options))

    components.append("# large affine matrix for low rank projection W_lowrank_-h")
    components.append("component name={0}_W_lowrank_-h type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, recurrent_projection_dim, 2 * (recurrent_projection_dim / 4), ng_affine_options))
    components.append("# low rank projection W_lowrank_h-h")
    components.append("component name={0}_W_lowrank_h-h type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, recurrent_projection_dim, recurrent_projection_dim / 4, ng_affine_options))

    components.append("# Defining the non-linearities")
    components.append("component name={0}_r type=SigmoidComponent dim={1}".format(name, recurrent_projection_dim))
    components.append("component name={0}_z type=SigmoidComponent dim={1}".format(name, recurrent_projection_dim))
    components.append("component name={0}_h_tilt type=TanhComponent dim={1}".format(name, recurrent_projection_dim))

    components.append("# Defining the hidden node computations")
    components.append("component name={0}_rh type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * recurrent_projection_dim, recurrent_projection_dim))
    components.append("component name={0}_h1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * recurrent_projection_dim, recurrent_projection_dim))
    components.append("component name={0}_h2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * recurrent_projection_dim, recurrent_projection_dim))
    components.append("component name={0}_h type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))

    components.append("# Defining fixed scale/bias component for (1 - z_t)")
    components.append("component name={0}_fixed_scale_minus_one type=FixedScaleComponent scales={1}".format(name, scale_minus_one_file))
    components.append("component name={0}_fixed_bias_one type=FixedBiasComponent bias={1}".format(name, bias_one_file))

    component_nodes.append("# large affine transform")
    component_nodes.append("component-node name={0}_large_affine component={0}_W_lowrank_-h input=IfDefined(Offset({0}_h_t, {1}))".format(name, gru_delay))
    
    component_nodes.append("# r_t")
    component_nodes.append("dim-range-node name={0}_r1_pre input-node={0}_large_affine dim-offset=0 dim={1}".format(name, recurrent_projection_dim / 4))
    component_nodes.append("component-node name={0}_r1 component={0}_W_r-xh input=Append({1}, {0}_r1_pre)".format(name, input_descriptor))
    component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r1".format(name))
   
    component_nodes.append("# z_t")
    component_nodes.append("dim-range-node name={0}_z1_pre input-node={0}_large_affine dim-offset={1} dim={2}".format(name, recurrent_projection_dim / 4, recurrent_projection_dim / 4))
    component_nodes.append("component-node name={0}_z1 component={0}_W_z-xh input=Append({1}, {0}_z1_pre)".format(name, input_descriptor))
    component_nodes.append("component-node name={0}_z_t component={0}_z input={0}_z1".format(name))

    component_nodes.append("# h_tilt_t")
    component_nodes.append("component-node name={0}_rh_t component={0}_rh input=Append({0}_r_t, IfDefined(Offset({0}_h_t, {1})))".format(name, gru_delay))
    component_nodes.append("component-node name={0}_h_tilt1_pre component={0}_W_lowrank_h-h input={0}_rh_t".format(name, gru_delay))
    component_nodes.append("component-node name={0}_h_tilt1 component={0}_W_h-xh input=Append({1}, {0}_h_tilt1_pre)".format(name, input_descriptor))
    component_nodes.append("component-node name={0}_h_tilt_t component={0}_h_tilt input={0}_h_tilt1".format(name))

    component_nodes.append("# The following two lines are to implement (1 - z_t)")
    component_nodes.append("component-node name={0}_minus_z_t component={0}_fixed_scale_minus_one input={0}_z_t".format(name))
    component_nodes.append("component-node name={0}_one_minus_z_t component={0}_fixed_bias_one input={0}_minus_z_t".format(name))

    component_nodes.append("# h_t") 
    component_nodes.append("component-node name={0}_h1_t component={0}_h1 input=Append({0}_z_t, {0}_h_tilt_t)".format(name))
    component_nodes.append("component-node name={0}_h2_t component={0}_h2 input=Append({0}_one_minus_z_t, IfDefined(Offset({0}_h_t, {1})))".format(name, gru_delay))
    
    components.append("# projection matrices : W-m; and nonlinearity transform : relu and renorm")
    components.append("component name={0}_W-m type=NaturalGradientAffineComponent input-dim={1} output-dim={2} bias-stddev=0".format(name, recurrent_projection_dim, non_recurrent_projection_dim))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1}".format(name, non_recurrent_projection_dim))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms=1.0".format(name, non_recurrent_projection_dim))

    component_nodes.append("# h_t and p_t")
    component_nodes.append("component-node name={0}_h_t component={0}_h input=Sum({0}_h1_t, {0}_h2_t)".format(name))
    component_nodes.append("component-node name={0}_p_t component={0}_W-m input={0}_h_t".format(name))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={0}_p_t".format(name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu".format(name))

    output_descriptor = '{0}_renorm'.format(name)
    output_dim = non_recurrent_projection_dim

    return {
            'descriptor': output_descriptor,
            'dimension':output_dim
            }


