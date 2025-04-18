from torch.functional import block_diag
from base_compiler import Compiler
import os
import numpy as np
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'
try:
    # from openvino.inference_engine import IENetwork, ExecutableNetwork, IECore
    # import openvino.inference_engine.ie_api
    from mo.main import emit_ir
except:
    print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW + ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
    exit(1)

# TODO: add parameter check mechnism
class VPU_compiler(Compiler):
    def __init__(self, output_dir='./model_pool/'):
        super(VPU_compiler, self).__init__(output_dir)

    def prepare_compile(self, **params):
        pass
    
    def compile(self, in_h, in_w, in_c, name='subnet'):
        inp_shape = [1, in_h, in_w, in_c]
        # converterd model with NCHW input format
        # cmd = "mo.py --data_type=FP16 --input_model={}.pb --input_shape=[1,{},{},3] --silent".format(self.subnet_name, inp_sz, inp_sz)
        # cmd = "mo_tf.py --data_type=FP16 --input_model=./{}/{}.pb --input_shape={} --silent --output_dir {} 1>/dev/null 2>&1".format(self.output_dir, name, str(inp_shape).replace(' ', ''), self.output_dir)
        cmd = "mo_tf.py --data_type=FP16 --input_model=./{}/{}.pb --input_shape={} --output_dir {} --silent ".format(self.output_dir, name, str(inp_shape).replace(' ', ''), self.output_dir)
        ret = os.system(cmd)
        assert(ret == 0), f"compile error ! {ret}"
        ret = format(ret, '016b')
        exit_code = ret[:8]
        signal_code = ret[8:]
        return [self.output_dir + name + '.xml', self.output_dir + name + '.bin']

    def finish_compile(self, **parmas):
        raise NotImplementedError

    @staticmethod
    def repr_shape(shape):
        # copy from https://github.com/mit-han-lab/once-for-all/blob/cfa0722d57e3a2391eb36b8cf613dd17ff7a32ae/ofa/tutorial/latency_table.py
        if isinstance(shape, (list, tuple)):
            return 'x'.join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def query(self, lut:dict, l_type: str, input_shape, output_shape, mid=None, ks=None, stride=None, id_skip=None,
              se=None, h_swish=None):
        infos = [l_type, 'input:%s' % self.repr_shape(input_shape), 'output:%s' % self.repr_shape(output_shape), ]

        if l_type in ('expanded_conv',):
            assert None not in (mid, ks, stride, id_skip, se, h_swish)
            infos += ['expand:%d' % mid, 'kernel:%d' % ks, 'stride:%d' % stride, 'idskip:%d' % id_skip,
                      'se:%d' % se, 'hs:%d' % h_swish]
        key = '-'.join(infos)
        return lut[key]

    def keras2pb(self, net_tf, input_shape:list=[None,160,160,3], save_name:str='subnet'):
        # TODO: move converter to a seperate module
        import tensorflow as tf
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        full_model = tf.function(lambda x : net_tf(x))
        # full_model = full_model.get_concrete_function(tf.TensorSpec(shape=[None, input_shape, input_shape, 3], dtype=tf.float32))
        full_model = full_model.get_concrete_function(tf.TensorSpec(shape=input_shape, dtype=tf.float32))
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                            logdir=self.output_dir,
                            name=save_name + '.pb',
                            as_text=False)
        # tf.keras.backend.clear_session()
        del full_model  
        del frozen_func   
        with tf.io.gfile.GFile(self.output_dir + '/' + save_name + '.pb', 'rb')as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())  
        cmd = 'rm ' + self.output_dir + '/' + save_name + '.pb'
        os.system(cmd)
        return graph_def   

    def graph_def_to_VPU_IR(self, graph_def, variables_values, **params):
        from mo.front.common.register_custom_ops import check_for_duplicates
        from mo.front.common.register_custom_ops import update_extractors_with_extensions
        from mo.front.extractor import restore_edges, extract_node_attrs, remove_control_dependency_inputs
        from mo.front.tf.extractor import get_tf_edges, tf_op_extractor, tf_op_extractors
        from mo.front.tf.loader import protobuf2nx
        from mo.pipeline.common import get_ir_version
        from mo.utils import class_registration
        import logging as log

        try:
            tf.compat.v1.import_graph_def(graph_def, name='')
        except:
            assert(0), ("TensorFlow post-processing of loaded model was unsuccessful. "
                        "This is an optional step that Model Optimizer performs for any input model but it is not usually "
                        "required for all models."
                        "It likely means that the original model is ill-formed. "
                        "Model Optimizer will continue converting this model.")

        log.debug("Number of nodes in graph_def: {}".format(len(graph_def.node))) 
        # print("op extractors before update: ", tf_op_extractors)
        update_extractors_with_extensions(tf_op_extractors)
        # print("op extractors after update: ", tf_op_extractors)
        argv = params['argv']
        # print("input shape ", argv.input_shape)
        try:
            graph = protobuf2nx(graph_def)
            graph.__setattr__('name', argv.model_name)
            # 'layout' parameter change may cause an issue in EltwiseInputReshape replacer
            # and convert_nhwc_to_nchw(graph)
            graph.graph['layout'] = 'NCHW' if argv.disable_nhwc_to_nchw else 'NHWC'
            graph.graph['cmd_params'] = argv
            graph.graph['fw'] = 'tf'
            graph.graph['ir_version'] = get_ir_version(argv)

            graph.graph['variables_values'] = variables_values
            del variables_values

            graph = restore_edges(graph, get_tf_edges)
            graph = remove_control_dependency_inputs(graph)
        except Exception as e:
            assert(0), (
                'Cannot pre-process TensorFlow graph after reading from model file "{}". ' \
                'File is corrupt or has unsupported format. Details: {}. ' +
                argv.model_name,
                str(e)
            )

        graph.check_empty_graph('protobuf2nx. It may happen due to problems with loaded model')
        extract_node_attrs(graph, lambda node: tf_op_extractor(node, check_for_duplicates(tf_op_extractors)))
        # --------------------------------- LOAD END ------------------------------------------------------

        class_registration.apply_replacements(graph, [
            class_registration.ClassType.FRONT_REPLACER,
            class_registration.ClassType.MIDDLE_REPLACER,
            class_registration.ClassType.BACK_REPLACER
        ])
        return graph

    def generate_block_with_key_given_inp_sz(self, input_sz:int=160, framework='tf'):
        '''generate dict idx for block ir'''
        from ofa.model_zoo import ofa_net
        from proxyless_nets_tf2 import convert_block_list
        from ofa.imagenet_codebase.networks.proxyless_nets import MobileInvertedResidualBlock
        import copy
        # TODO: move framework into configuration file or params
        assert(framework == 'tf'), "Temporarilly tf is supported."
        blocks_dict = {}
        supernet = ofa_net('ofa_proxyless_d234_e346_k357_w1.3', pretrained=True)
        # fixed operation given input size : first_conv, first_block and final classifier.
        # fisrt conv
        out_sz = (input_sz + 1) // 2
        first_conv_key_info = ['first_conv', 'input:%s'% self.repr_shape([input_sz, input_sz, 3]), 'output:%s' % self.repr_shape([out_sz, out_sz, supernet.first_conv.out_channels])]
        first_conv_key = '-'.join(first_conv_key_info)
        first_conv_block = self.keras2pb(convert_block_list([supernet.first_conv], (input_sz, input_sz, 3)), input_shape=[None, input_sz, input_sz, 3])
        assert(not first_conv_key in blocks_dict), 'Added repeated block IR. {}'.format(first_conv_key)
        blocks_dict[first_conv_key] = {'block':first_conv_block, 'input_shape' : (input_sz, input_sz, 3), 'data_format':('HWC')}
        print("\t added block IR : ", first_conv_key)

        # first block
        input_sz = (input_sz+1)//2
        first_block = supernet.blocks[0]
        mb_conv = first_block.mobile_inverted_conv
        assert(mb_conv is not None), "mb_conv should not be None"
        shortcut = first_block.shortcut
        if shortcut is None:
            idskip = 0
        else:
            idskip = 1
        out_sz = int((input_sz - 1) / mb_conv.stride + 1)
        first_block_key_info = ['expanded_conv', 'input:%s'% self.repr_shape([input_sz, input_sz, mb_conv.in_channels]), 'output:%s' % self.repr_shape([out_sz, out_sz, mb_conv.out_channels]), 
                                'expand:%d' % mb_conv.depth_conv.conv.in_channels, 'kernel:%d' % mb_conv.kernel_size, 'stride:%d' % mb_conv.stride, 'idskip:%d' % idskip,
                                'se:%d' % (1 if mb_conv.use_se else 0), 'hs:%d' % (1 if mb_conv.act_func == 'h_swish' else 0)]
        print(first_block_key_info)
        first_block_key = '-'.join(first_block_key_info)
        first_block = self.keras2pb(convert_block_list([first_block], (input_sz, input_sz, mb_conv.in_channels)), input_shape=[None, input_sz, input_sz, mb_conv.in_channels])
        assert(not first_block_key in blocks_dict), 'Added repeated block IR. {}'.format(first_block_key)
        # TODO : put these attribute to global resource manager ?
        blocks_dict[first_block_key] =  {'block':first_block, 'input_shape' : (input_sz, input_sz, mb_conv.in_channels), 'data_format':('HWC')}
        print("\t added block IR : ", first_block_key)

        # blocks[1:]
        kernel_sz_choice = [3,5,7]
        expand_ratio_choice = [3,4,6]
        input_channel = supernet.blocks[0].mobile_inverted_conv.out_channels
        for b in supernet.blocks[1:]:
            input_sz = out_sz
            # try every kernel size and active_expand_ratio.
            for ksz in kernel_sz_choice:
                for er in expand_ratio_choice:
                    b.mobile_inverted_conv.active_kernel_size = ksz
                    b.mobile_inverted_conv.active_expand_ratio = er
                    preserve_weight = True
                    block_sample = MobileInvertedResidualBlock(
                        b.mobile_inverted_conv.get_active_subnet(input_channel, preserve_weight),
                        copy.deepcopy(b.shortcut)
                    )
                    mb_conv = block_sample.mobile_inverted_conv
                    assert(mb_conv is not None), "mb_conv should not be None"
                    shortcut = block_sample.shortcut
                    if shortcut is None:
                        idskip = 0
                    else:
                        idskip = 1
                    out_sz = int((input_sz - 1) / mb_conv.stride + 1)
                    block_key_info = ['expanded_conv', 'input:%s'% self.repr_shape([input_sz, input_sz, mb_conv.in_channels]), 'output:%s' % self.repr_shape([out_sz, out_sz, mb_conv.out_channels]), 
                                        'expand:%d' % mb_conv.depth_conv.conv.in_channels, 'kernel:%d' % mb_conv.kernel_size, 'stride:%d' % mb_conv.stride, 'idskip:%d' % idskip,
                                        'se:%d' % (1 if mb_conv.use_se else 0), 'hs:%d' % (1 if mb_conv.act_func == 'h_swish' else 0)]
                    block_key = '-'.join(block_key_info)
                    print(f"block key {block_key} \n kernel size {ksz}, expand_ratio {er}")
                    if block_key in blocks_dict:
                        continue
                    block_sample = self.keras2pb(convert_block_list([block_sample], (input_sz, input_sz, mb_conv.in_channels)), input_shape=[None, input_sz, input_sz, mb_conv.in_channels])
                    assert(not block_key in blocks_dict), 'Added repeated block IR.'.format(block_key)
                    # blocks_dict[block_key] =  block_sample
                    blocks_dict[block_key] = {'block':block_sample, 'input_shape' : (input_sz, input_sz, mb_conv.in_channels), 'data_format':('HWC')}
                    print("\t added block IR : ", block_key)
            input_channel = mb_conv.out_channels
        # feature mix layer and classifier
        input_sz = out_sz
        block_list = [supernet.feature_mix_layer, supernet.classifier]
        classifier_key_info = ['classifier', 'input:%s'% self.repr_shape([input_sz, input_sz, block_list[0].in_channels]), 'output:%s'% self.repr_shape([supernet.classifier.out_features])]
        classifier_key = '-'.join(classifier_key_info)
        classifier = self.keras2pb(convert_block_list(block_list, (input_sz, input_sz, block_list[0].in_channels)), input_shape=[None, input_sz, input_sz, block_list[0].in_channels])
        assert(not classifier_key in blocks_dict), 'Added repeated block IR. {}'.format(classifier_key)
        blocks_dict[classifier_key] = {'block':classifier, 'input_shape' : (input_sz, input_sz, block_list[0].in_channels), 'data_format':('HWC')}
        print("\t added block IR : ", classifier_key)
        print("block number ", len(blocks_dict.keys()))
        return blocks_dict

    def block_ir_generator(self, block, framework:str, **params):
        '''generate block ir with respect to the block'''
        argv = params['argv']
        shape = params['input_shape']
        argv.input_shape = str([1, shape[0], shape[1], shape[2]])
        argv.placeholder_shapes = np.array([1, shape[0], shape[1], shape[2]], dtype=np.int64)
        params['argv'] = argv
        graph_def = block
        nodes_to_clear_device = graph_def.node if isinstance(graph_def, tf.compat.v1.GraphDef) else graph_def.graph_def.node
        for node in nodes_to_clear_device:
            node.device = ""
        variables_values = {} 
        graph = self.graph_def_to_VPU_IR(graph_def, variables_values, **params)
        return graph
        # raise NotImplementedError

    def prepare_ir_env(self):
        # TODO: add support for prepare_ir, move argv to configuration file or params
        from mo.utils.versions_checker import check_requirements
        from mo.utils.cli_parser import get_tf_cli_parser, append_exp_keys_to_namespace, get_mean_scale_dictionary, parse_tuple_pairs, get_freeze_placeholder_values
        from mo.utils.logger import init_logger
        # configure hyperparameters
        argv = get_tf_cli_parser().parse_args()
        append_exp_keys_to_namespace(argv)
        argv.framework = 'tf'
        argv.model_name = 'subnet'
        # cmd params
        argv.data_type = "FP16"
        argv.output_dir = self.output_dir
        argv.silent = True
        argv.input_model = './{}/{}.pb'.format(self.output_dir, argv.model_name)
        argv.placeholder_data_types = {}
        mean_values = parse_tuple_pairs(argv.mean_values)
        scale_values = parse_tuple_pairs(argv.scale_values)
        mean_scale = get_mean_scale_dictionary(mean_values, scale_values, argv.input)
        argv.mean_scale_values = mean_scale
        argv.freeze_placeholder_with_value, argv.input = get_freeze_placeholder_values(argv.input,
                                                                argv.freeze_placeholder_with_value)
        init_logger(argv.log_level.upper(), argv.silent)
        # ret_code = check_requirements(framework=argv.framework)
        # assert(ret_code == 0), 'check_requirements exit with return code {}'.format(ret_code)
        
        if hasattr(argv, 'extensions') and argv.extensions and argv.extensions != '':
            extensions = argv.extensions.split(',')
        else:
            extensions = None
        # TODO : move to outer loop?
        from mo.front.tf.register_custom_ops import get_front_classes
        from mo.utils import import_extensions
        # print("extensions are ", extensions)
        # print("framework is ", argv.framework)
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)
        return argv

    def generate_block_IR_given_inp_sz(self, block_dict_generator:callable, block_ir_generator:callable, input_size:int=160, framework:str='tf', **params):
        block_irs = {}
        block_dict = block_dict_generator(input_size, framework)
        
        for k, b in block_dict.items():
            if not k in block_irs:
                params['input_shape'] = b['input_shape']
                # print("argv in params ", params['argv'])
                ir = block_ir_generator(b['block'], framework, **params)
                block_irs[k] = ir
        return block_irs
            
    def build_IR_LUT(self):
        import dill
        lut = {}
        # input_sizes = (160, 176, 192, 208, 224)
        input_sizes = (160,)
        framework = 'tf'
        argv = self.prepare_ir_env()
        for inp_sz in input_sizes:
            block_lut = self.generate_block_IR_given_inp_sz(self.generate_block_with_key_given_inp_sz, self.block_ir_generator, inp_sz, framework, argv=argv)
            lut.update(block_lut)
        with open('blocks_IR.pkl', 'wb') as f:
            dill.dump(lut, f)

    def assemble_block_IR(self, subnet1, subnet2):
        """ concat subnet2 to subnet1 """
        subnet1_out_edge = list(subnet1.in_edges('Identity/sink_port_0', data=True))
        subnet1.remove_node('Identity/sink_port_0')
        assert(len(subnet1_out_edge) == 1), "only one output is supported now."
        out_edge = subnet1_out_edge[0]
        in_edge_list = list(subnet2.edges('x/Output_0/Data_', data=True))
        subnet2.remove_node('x')
        subnet2.remove_node('x/Output_0/Data_')

        # add edges & nodes from subnet2 to subnet1, rename node names.
        node_list = list(subnet2.nodes(data=True))
        # edge_list = list(subnet2.edges(data=True))
        # add_node_list = []
        add_edge_list = []
        cnt = 0
        for node in node_list:
            node_name = node[0]
            # get node with same names.
            if node_name in subnet1.nodes:
                print("node already exist, renaming..", node_name, node)
                # modify edge attrs.
                in_edges = list(subnet2.in_edges(node_name, data=True))
                out_edges = list(subnet2.out_edges(node_name, data=True))
                while (node_name + '_' + str(cnt) in subnet1.nodes):
                    cnt += 1
                new_node_name = node_name + '_' + str(cnt)
                node[1]['name'] = new_node_name
                # (node_id, node.attr)
                new_node = (new_node_name, node[1])
                subnet2.remove_node(node_name)
                subnet2.add_nodes_from([new_node])
                for in_e in in_edges:
                    new_edge = (in_e[0], new_node_name, in_e[2])
                    add_edge_list.append(new_edge)
                for out_e in out_edges:
                    new_edge = (new_node_name, out_e[1], out_e[2])
                    add_edge_list.append(new_edge)
                subnet2.add_edges_from(add_edge_list)
                add_edge_list = []
        subnet1.add_nodes_from(list(subnet2.nodes(data=True)))
        subnet1.add_edges_from(list(subnet2.edges(data=True)))

        # add the edge to concat the two subgraph
        source_node = out_edge[0]
        for e in in_edge_list:
            end_node = e[1]
            attr = e[2]
            new_edge = (source_node, end_node, attr)
            subnet1.add_edges_from([new_edge])   
        return subnet1   

    def compile_LUT(self, net, in_h:int, in_w:int, in_c:int, name='subnet', **params):
        assert(in_h == in_w), 'Input should have equal width and height.'
        from mo.utils.cli_parser import get_tf_cli_parser
        import time
        try:
            from mo.main import emit_ir
        except:
            print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW + ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
            exit(1)
        start_time = time.perf_counter()
        cli_parser = get_tf_cli_parser()
        image_size = in_h
        import dill
        with open('blocks_IR.pkl', 'rb') as f:
            lut = dill.load(f)
        # first conv
        net_ir = self.query(lut, 'Conv', [image_size, image_size, 3], [(image_size + 1) // 2, (image_size + 1) // 2, net.first_conv.out_channels])

        # blocks
        fsize = (image_size + 1) // 2
        for block in net.blocks:
            mb_conv = block.mobile_inverted_conv
            shortcut = block.shortcut

            if mb_conv is None:
                continue
            if shortcut is None:
                idskip = 0
            else:
                idskip = 1
            out_fz = int((fsize - 1) / mb_conv.stride + 1)
            block_ir = self.query(lut,
                'expanded_conv', [fsize, fsize, mb_conv.in_channels], [out_fz, out_fz, mb_conv.out_channels],
                mid=mb_conv.depth_conv.conv.in_channels, ks=mb_conv.kernel_size, stride=mb_conv.stride, id_skip=idskip,
                se=1 if mb_conv.use_se else 0, h_swish=1 if mb_conv.act_func == 'h_swish' else 0,
            )
            fsize = out_fz
            net_ir = self.assemble_block_IR(net_ir, block_ir)
        # feature mix layer and classifier
        block_ir = self.query('classifier', [fsize, fsize, net.blocks[-1].out_channels],
                            [net.classifier.out_features])
        net_ir = self.assemble_block_IR(net_ir, block_ir)

        # TODO: move argv into params
        argv = net_ir.graph['cmd_params']
        # emit_ir need argv.[output_dir, framework, model_name,]
        ret_res = emit_ir(net_ir, argv)

        if ret_res != 0:
            return ret_res

        elapsed_time = time.perf_counter() - start_time
        print('[ SUCCESS ] Total execution time: {:.2f} seconds. '.format(elapsed_time.total_seconds()))

        return ret_res

def get_model_name(path_input_model: str) -> str:
    """
    Deduces model name by a given path to the input model
    Args:
        path_input_model: path to the input model

    Returns:
        name of the output IR
    """
    parsed_name, extension = os.path.splitext(os.path.basename(path_input_model))
    return 'model' if parsed_name.startswith('.') or len(parsed_name) == 0 else parsed_name


if __name__ == '__main__':
    compiler = VPU_compiler()
    compiler.build_IR_LUT()