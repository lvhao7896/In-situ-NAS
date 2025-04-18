from copy import copy
import tensorflow as tf
# from tensorflow.python.keras.backend import depthwise_conv2d
from torch import nn
from ofa.imagenet_codebase.networks.proxyless_nets import MobileInvertedResidualBlock
import ofa

def init_ConvLayer_tf2(convlayer_torch, input):
    shape= convlayer_torch.weight.shape
    convlayer_torch = convlayer_torch
    pads = convlayer_torch.padding
    padding = None
    (out_c, in_c, k_h, k_w) = shape
    if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
        padding = (pads[0], pads[1])
    elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
        padding = (pads[0], pads[1], pads[2], pads[3])

    groups = convlayer_torch.groups
    stride = convlayer_torch.stride
    weights_list = []
    conv_torch = convlayer_torch.eval()
    has_bias = False if convlayer_torch.bias==None else True
    if groups !=1 and in_c == 1: # depthwise conv
        # from (cout, cin, ks, ks) to (ks, ks, cin, cout)
        weight = conv_torch.weight.data.numpy().transpose(2,3,0,1).copy()
        weights_list = [weight] 
    elif groups == 1:
        weight = conv_torch.weight.data.numpy().transpose(2,3,1,0).copy()
        weights_list = [weight]
    else:
        print("groups ", groups)
        print((out_c, in_c, k_h, k_w))
        raise NotImplementedError
    if conv_torch.bias is not None:
        bias = conv_torch.bias.data.numpy().copy()
        weights_list = [weight, bias]  
    conv_tf_out = ConvLayer_tf2_forward(shape, padding, groups, stride, has_bias, input, weights_list)
    # conv_tf.set_weights(weights_list)
    return conv_tf_out

def ConvLayer_tf2_forward(shape, padding, groups, stride, has_bias, input, weights_list):
    (out_c, in_c, k_h, k_w) = shape
    # self.shape = shape
    # # self.weights_list = weights_list
    # self.padding = padding
    # self.groups = groups
    # self.stride = stride
    # self.padding_layer = None
    # self.has_bias = has_bias
    padding_layer = None
    if padding is not None:
        padding_layer = tf.keras.layers.ZeroPadding2D(padding=padding)

    if groups !=1 and in_c == 1:
        # weights = self.get_depth_conv_weights(convlayer_torch)
        conv = tf.keras.layers.DepthwiseConv2D((k_h, k_w), stride, padding='valid', use_bias=has_bias, weights=weights_list, trainable=False)
        # conv = tf.keras.layers.DepthwiseConv2D((k_h, k_w), stride, padding='valid', use_bias=has_bias,  trainable=False)
    elif groups == 1:
        # weights = self.get_conv_weights(convlayer_torch)
        conv = tf.keras.layers.Conv2D(out_c, (k_h, k_w), stride, padding='valid', groups=groups, use_bias=has_bias, weights=weights_list, trainable=False)
        # conv = tf.keras.layers.Conv2D(out_c, (k_h, k_w), stride, padding='valid', groups=groups, use_bias=has_bias, trainable=False)
    else:
        print("groups ", groups)
        print((out_c, in_c, k_h, k_w))
        raise NotImplementedError
    if padding_layer is not None:
        input = padding_layer(input)
    out = conv(input)
    return out

def init_Conv_BN_act_tf2(conv_torch, bn_torch, act_func, input):
    assert isinstance(conv_torch, nn.Conv2d)
    assert isinstance(bn_torch, nn.BatchNorm2d), 'Unsupported layer {}, BN needed in Conv_BN_act_tf2'.format(type(bn_torch))

    bn_torch = bn_torch.eval()
    weight = bn_torch.weight.data.numpy().copy()
    bias = bn_torch.bias.data.numpy().copy()
    running_mean = bn_torch.running_mean.data.numpy().copy()
    running_var = bn_torch.running_var.data.numpy().copy()
    bn_weights = [weight, bias, running_mean, running_var]
    bn = tf.keras.layers.BatchNormalization(trainable=False, momentum=0.9, epsilon=1e-03, weights=bn_weights)

    supported_acts = ['relu6', 'linear']
    assert act_func in supported_acts, 'Unsportted act function {}.'.format(act_func)
    if act_func == 'relu6':
        act = tf.keras.layers.ReLU(max_value=6)
    else:
        act = None

    conv_out = init_ConvLayer_tf2(conv_torch, input)
    x = bn(conv_out)
    if act:
        x = act(x)
    return x

# class Conv_BN_act_tf2(tf.keras.layers.Layer):
# # class Conv_BN_act_tf2(tf.keras.Model):
#     # def __init__(self, conv, bn_weights:list, act_func:str,**kwargs):
#     def __init__(self, conv,  act_func:str,**kwargs):
#         super(Conv_BN_act_tf2, self).__init__(**kwargs)
#         self.conv = conv
#         # self.bn_weights = bn_weights
#         self.act_func = act_func
#         # self.bn = tf.keras.layers.BatchNormalization(trainable=False, momentum=0.9, epsilon=1e-03, weights=bn_weights)
#         self.bn = tf.keras.layers.BatchNormalization(trainable=False, momentum=0.9, epsilon=1e-03)

#         supported_acts = ['relu6', 'linear']
#         assert act_func in supported_acts, 'Unsportted act function {}.'.format(act_func)
#         if act_func == 'relu6':
#             self.act = tf.keras.layers.ReLU(max_value=6)
#         else:
#             self.act = None

#     def call(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         if self.act:
#             x = self.act(x)
#         return x

#     def get_config(self):
#         config = {
#             "conv":self.conv,
#             # "bn_weights":self.bn_weights,
#             "act_func":self.act_func,
#         }
#         base_config = super(Conv_BN_act_tf2, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

def init_Depthwise_Conv_tf2(dc_torch, input):
    depth_conv_torch, bn_torch, act_torch = dc_torch
    assert isinstance(bn_torch, nn.BatchNorm2d), 'BN needed in depthwise conv'
    assert isinstance(act_torch, nn.ReLU6), 'depthwise conv only support relu6'
    out = init_Conv_BN_act_tf2(depth_conv_torch, bn_torch, 'relu6', input)
    return out
    # return Depthwise_Conv_tf2(block)

# class Depthwise_Conv_tf2(tf.keras.layers.Layer):
#     def __init__(self, block,**kwargs):
#         super(Depthwise_Conv_tf2, self).__init__(**kwargs)
#         self.block = block
#         return block

#     def call(self, x):
#         return self.block(x)
    
#     def get_config(self):
#         config = {
#             "block":self.block
#         }
#         base_config = super(Depthwise_Conv_tf2, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

def init_Point_Conv_tf2(pc_torch, input):
    point_conv_torch, bn_torch = pc_torch
    assert isinstance(bn_torch, nn.BatchNorm2d), 'BN needed in depthwise conv'
    assert point_conv_torch.weight.shape[2:] == (1,1), 'kernel size should be 1 in point conv'
    out = init_Conv_BN_act_tf2(point_conv_torch, bn_torch, 'linear', input)
    return out
    # return Point_Conv_tf2(block)

# class Point_Conv_tf2(tf.keras.layers.Layer):
#     def __init__(self, block,**kwargs):
#         super(Point_Conv_tf2, self).__init__(**kwargs)
#         self.block = block
#         return block
    
#     def call(self, x):
#         return self.block(x)

#     def get_config(self):
#         config = {
#             "block":self.block
#         }
#         base_config = super(Point_Conv_tf2, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

# def init_IdentityLayer_tf2(shortcut):
#     assert isinstance(shortcut, IdentityLayer)
#     return IdentityLayer_tf2()

# class IdentityLayer_tf2(tf.keras.layers.Layer):
#     def __init__(self,**kwargs):
#         super(IdentityLayer_tf2, self).__init__(**kwargs)

#     def call(self, x):
#         return x

#     def get_config(self):
#         config = {}
#         base_config = super(IdentityLayer_tf2, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

def init_MBInvertedConvLayer_tf2(mb_inveted_conv, x):
    assert mb_inveted_conv is not None
    assert mb_inveted_conv.use_se == False, 'Not supported for SE Module'
    if mb_inveted_conv.expand_ratio == 1:
        inverted_bottleneck_out = None
    else:
        conv, bn, act = mb_inveted_conv.inverted_bottleneck
        assert(isinstance(act, nn.ReLU6)), 'Activation function only support relu6. ' + str(type(act))
        x = init_Conv_BN_act_tf2(conv, bn, 'relu6', x)

    # if inverted_bottleneck_out is not None:
    #     depth_conv_out = init_Depthwise_Conv_tf2(mb_inveted_conv.depth_conv, x)
    # else:
    depth_conv_out = init_Depthwise_Conv_tf2(mb_inveted_conv.depth_conv, x)
    point_linear_out = init_Point_Conv_tf2(mb_inveted_conv.point_linear, depth_conv_out) 
    return point_linear_out 
    # return MBInvertedConvLayer_tf2(inverted_bottleneck, depth_conv, point_linear)

def init_MobileInvertedResidualBlock_tf2(mobile_inverted_conv, shortcut, input):
    mobile_inverted_conv_out = init_MBInvertedConvLayer_tf2(mobile_inverted_conv, input)
    # if shortcut is not None:
    #     has_shortcut = True
    # else:
    #     has_shortcut = False

    if mobile_inverted_conv is None:
        res = input
    elif shortcut is None:
        res = mobile_inverted_conv_out
    else:
        res = tf.keras.layers.Add()([mobile_inverted_conv_out, input])
    return res
    # return MobileInvertedResidualBlock_tf2(mobile_inverted_conv, has_shortcut)

# class MobileInvertedResidualBlock_tf2(tf.keras.layers.Layer):
#     def __init__(self, mobile_inverted_conv, has_shortcut,**kwargs):
#         super(MobileInvertedResidualBlock_tf2, self).__init__(**kwargs)
#         self.mobile_inverted_conv = mobile_inverted_conv
#         self.has_shortcut = has_shortcut
#         block = tf.keras.Sequential()

#         if self.mobile_inverted_conv is None:
#             res = x
#         elif self.has_shortcut is None:
#             block.add(self.mobile_inverted_conv)
#         else:
#             res = tf.keras.layers.Add()([self.mobile_inverted_conv(x), x])       

#     def call(self, x):
#         if self.mobile_inverted_conv is None:
#             res = x
#         elif self.has_shortcut is None:
#             res = self.mobile_inverted_conv(x)
#         else:
#             res = tf.keras.layers.Add()([self.mobile_inverted_conv(x), x])
#         return res

#     def get_config(self):
#         config = {
#             "mobile_inverted_conv":self.mobile_inverted_conv,
#             "shortcut":self.shortcut
#         }
#         base_config = super(MobileInvertedResidualBlock_tf2, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

def init_FCLayer_tf2(fc_layer_torch, input):
    fc_torch = fc_layer_torch.eval()
    out_filters = fc_layer_torch.out_features
    has_bias = False if fc_layer_torch.bias==None else True
    # from (cout, cin) to (cin, cout)
    weight = fc_torch.weight.data.numpy().transpose(1,0).copy()
    weights_list = [weight]
    if fc_torch.bias is not None:
        bias = fc_torch.bias.data.numpy().copy()
        weights_list = [weight, bias]
    # return FCLayer_tf2(out_filters, weights_list)
    fc = tf.keras.layers.Dense(out_filters, use_bias=has_bias, weights=weights_list)
    fc_out = fc(input)
    return fc_out

class FCLayer_tf2(tf.keras.layers.Layer):
    # def __init__(self, out_filters, weights_list,**kwargs):
    def __init__(self, out_filters, has_bias, **kwargs):
        super(FCLayer_tf2, self).__init__(**kwargs)
        self.out_filters = out_filters
        self.has_bias = has_bias
        # self.weights_list = weights_list
        # self.fc = tf.keras.layers.Dense(out_filters, use_bias=has_bias, weights=weights_list)
        self.fc = tf.keras.layers.Dense(out_filters, use_bias=has_bias)

    def call(self, x):
        x = self.fc(x)
        return x

    # def get_config(self):
    #     config = {
    #         "out_filters":self.out_filters,
    #         "has_bias":self.has_bias,
    #         # "weights_list":self.weights_list,
    #     }
    #     base_config = super(FCLayer_tf2, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

def convert_block(block, input):
    if isinstance(block, ofa.layers.ConvLayer):
        out = init_Conv_BN_act_tf2(block.conv, block.bn, 'relu6', input)
        return out
    elif isinstance(block, MobileInvertedResidualBlock):
        out = init_MobileInvertedResidualBlock_tf2(block.mobile_inverted_conv, block.shortcut, input)
        return out
    elif isinstance(block, ofa.layers.LinearLayer):
        avg_layer = tf.keras.layers.GlobalAveragePooling2D()
        out = avg_layer(input)
        out = init_FCLayer_tf2(block.linear, out)
        return out
    else:
        print(type(block), ' not implemented')
        raise NotImplementedError

def convert_block_list(blocks, inp_sz):
    input = tf.keras.layers.Input(shape=inp_sz)
    x = input
    for block in blocks:
        x = convert_block(block, x)
    return tf.keras.models.Model(input, x)

class ProxylessNASNets_tf2(tf.keras.layers.Layer):
    def __init__(self, first_conv, blocks, feature_mix_layer, classifier, inp_sz):
        super(ProxylessNASNets_tf2, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape = inp_sz)
        self.first_conv = init_Conv_BN_act_tf2(first_conv.conv, first_conv.bn, 'relu6')
        self.blocks = [init_MobileInvertedResidualBlock_tf2(x.mobile_inverted_conv, x.shortcut) for x in blocks]
        self.feature_mix_layer = init_Conv_BN_act_tf2(feature_mix_layer.conv, feature_mix_layer.bn, 'relu6')
        # self.padding_layer = tf.keras.layers.ZeroPadding2D(padding=((0,0),(1,1)))
        self.avg_layer = tf.keras.layers.GlobalAveragePooling2D()
        # self.avg_layer = tf.keras.layers.AveragePooling2D(pool_size=(5,5))
        # self.reshape_layer = tf.keras.layers.Reshape([-1])
        self.classifier = init_FCLayer_tf2(classifier.linear)
        
        self.model = tf.keras.Sequential()
        self.model.add(self.input_layer)
        self.model.add(self.first_conv)
        # self.model.add(tf.keras.Sequential(self.blocks))
        for block in self.blocks:
            self.model.add(block)
        self.model.add(self.feature_mix_layer)
        # self.model.add(self.padding_layer)
        self.model.add(self.avg_layer)
        # self.model.add(self.reshape_layer)
        self.model.add(self.classifier)

    def call(self, x):
        return self.model(x)
