import os
import tensorflow as tf
import time
from compilers.base_compiler import Compiler
# from  proxyless_nets_tf2 import  custom_objects
class Compiler:
    def __init__(self, output_dir='./model_pool/'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def prepare_compile(self, **params):
        raise NotImplementedError

    def compile(self, in_h, in_w, in_c, name='subnet'):
        raise NotImplementedError
    
    def finish_compile(self, **parmas):
        raise NotImplementedError


class TPU_compiler(Compiler):
    def __init__(self, output_dir='./model_pool/'):
        super(TPU_compiler, self).__init__(output_dir)
 
    def compile(self, in_h, in_w, in_c, name='subnet', silent=True):
        inp_shape = [None, in_h, in_w, in_c]
        if silent :
            os.system('edgetpu_compiler ./{}/{}.tflite  --out_dir {} 1>/dev/null 2>&1'.format(self.output_dir, name, self.output_dir))
        else :
            os.system('edgetpu_compiler ./{}/{}.tflite  --out_dir {}'.format(self.output_dir, name, self.output_dir))
        return [self.output_dir + name + '_edgetpu.tflite']

class DPU_compiler(Compiler):
    def __init__(self, output_dir='./model_pool/'):
        super(TPU_compiler, self).__init__(output_dir)

    def compile(self, in_h, in_w, in_c, name='subnet'):
        try:
            # quantize first
            shape_list = ['?', str(in_h), str(in_w), str(in_c)]
            inp_shape = ','.join(shape_list)
            quantize_cmd = 'vai_q_tensorflow '
            quantize_arg = 'quantize'
            quantize_flags = [
                quantize_cmd,
                quantize_arg, 
                'input_frozen_graph={}/{}.pb'.format(self.output_dir, name),
                '--input_fn={}'.format('prepare_pb.calib_input'),
                '--output_dir={}'.format(self.output_dir),
                '--input_nodes=input0_1',
                '--input_shapes='+inp_shape,
                '--output_nodes=output_1/BiasAdd',
                '--method=1', 
                '--skip_check=1',
                '--calib_iter=1',
                '1>/dev/null 2>&1'
            ]
            os.system(' '.join(quantize_flags))
            # compile   
            compile_cmd = 'vai_c_tensorflow'
            compile_flags = [
                compile_cmd,
                '--frozen_pb {}/{}.pb'.format(self.output_dir, name),
                '--arch Ultra96.json',
                '--output_dir {}'.format(self.output_dir),
                'net_name {}'.format(name),
                '1>/dev/null 2>&1'
            ]
            os.system(' '.join(compile_flags))
        except Exception as e:
            print("TPU compiler Error !", e)
        return [self.output_dir + name + '.elf']

class ARM_compiler(Compiler):
    def __init__(self, output_dir='./model_pool/'):
        super(ARM_compiler, self).__init__(output_dir)
        self.input_format = 'NCHW'

    def compile(self, in_h, in_w, in_c, name='subnet'):
        try:
            arch = 'arm64'
            target = 'llvm -mtriple=%s-linux-android' % arch
            target_host = None
            try:
                import json
                import logging
                logging.basicConfig(level=logging.CRITICAL)
                import tvm
                from tvm import relay
                from tvm.contrib import ndk
                assert tvm.runtime.enabled("rpc")
                ndk_cc_path = '/opt/android-toolchain-arm64/bin/aarch64-linux-android-g++'
                # ndk_cc_path = '/Users/blacktraker/Library/Android/sdk/ndk-bundle/build/tools/tmp_toolchain/bin/aarch64-linux-android-g++'
                os.environ.setdefault("TVM_NDK_CC", ndk_cc_path)
            except Exception as e:
                print("Import Error ! ", e)
                exit(0)
            load_start = time.perf_counter()
            # print("start load")
            model = tf.keras.models.load_model(self.output_dir + name + '.h5')
            # print(model.summary())
            # print(model.inputs)
            # print("Load done")
            load_end = time.perf_counter()
            assert(len(model.input_names) == 1), 'Only support one input.'
            input_name = model.input_names[0] # assume only one input
            # print("input name : ", input_name)
            inp_shape = (1, in_c, in_h, in_w)
            shape_dict = {input_name : inp_shape}
            compile_start = time.perf_counter()
            mod, params = relay.frontend.from_keras(model, shape_dict)
            with tvm.transform.PassContext(opt_level=3):
                graph, lib, params = relay.build(
                    mod, target=target,
                    params=params, target_host = target_host)
            lib_fname = self.output_dir + name + '.so'
            fcompile = ndk.create_shared 
            lib.export_library(lib_fname, fcompile)
            graph_file = self.output_dir + name + '.json'
            with open(graph_file, 'w') as outfile:
                json.dump(graph, outfile)

            del model
            compile_end = time.perf_counter()
            # print(f"[ARM compiler] load time : {load_end-load_start}, compile time : {compile_end - compile_start} total : {compile_end-load_start}")
            return [graph_file, lib_fname]
        except Exception as e:
            print("Compiler Error ! ", e)



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
        cmd = "mo_tf.py --data_type=FP16 --input_model=./{}/{}.pb --input_shape={} --output_dir {} ".format(self.output_dir, name, str(inp_shape).replace(' ', ''), self.output_dir)
        ret = os.system(cmd)
        assert(ret == 0), f"compile error ! {ret}"
        ret = format(ret, '016b')
        exit_code = ret[:8]
        signal_code = ret[8:]
        return [self.output_dir + name + '.xml', self.output_dir + name + '.bin']

    def finish_compile(self, **parmas):
        raise NotImplementedError

    def generate_block_IR(self, input_size=160):
        block_keys = [
            
        ]

    def build_IR_LUT(self, ):
        import dill
        lut = {}
        input_sizes = (160, 176, 192, 208, 224)

        for inp_sz in input_sizes:
            block_lut = self.generate_block_IR(inp_sz)
            lut.update(block_lut)
        with open('blocks_IR.pkl', 'rb') as f:
            dill.dump(lut, f)


    def query(self, lut:dict, l_type: str, input_shape, output_shape, mid=None, ks=None, stride=None, id_skip=None,
              se=None, h_swish=None):
        infos = [l_type, 'input:%s' % self.repr_shape(input_shape), 'output:%s' % self.repr_shape(output_shape), ]

        if l_type in ('expanded_conv',):
            assert None not in (mid, ks, stride, id_skip, se, h_swish)
            infos += ['expand:%d' % mid, 'kernel:%d' % ks, 'stride:%d' % stride, 'idskip:%d' % id_skip,
                      'se:%d' % se, 'hs:%d' % h_swish]
        key = '-'.join(infos)
        return lut[key]

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
        start_time = time.perf_counter()
        cli_parser = get_tf_cli_parser()
        inp_shape = [1, in_h, in_w, in_c]
        argv = cli_parser.parse_args()
        argv.framework = 'tf'
        argv.data_type = 'FP16'
        argv.input_model = './{}/{}.pb'.format(self.output_dir, name)
        argv.input_shape = str(inp_shape).replace(' ', '')
        argv.output_dir = self.output_dir
        argv.silent = True
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
        # final expand layer
        block_ir = self.query(
            'Conv_1', [fsize, fsize, net.final_expand_layer.in_channels],
            [fsize, fsize, net.final_expand_layer.out_channels],
        )
        net_ir = self.assemble_block_IR(net_ir, block_ir)

        # global average pooling
        block_ir = self.query(
            'AvgPool2D', [fsize, fsize, net.final_expand_layer.out_channels],
            [1, 1, net.final_expand_layer.out_channels],
        )
        net_ir = self.assemble_block_IR(net_ir, block_ir)

        # feature mix layer
        block_ir = self.query(
            'Conv_2', [1, 1, net.feature_mix_layer.in_channels],
            [1, 1, net.feature_mix_layer.out_channels]
        )
        net_ir = self.assemble_block_IR(net_ir, block_ir)

        # classifier
        block_ir = self.query(
            'Logits', [1, 1, net.classifier.in_features], [net.classifier.out_features]
        )
        net_ir = self.assemble_block_IR(net_ir, block_ir)

        ret_res = emit_ir(net_ir, argv)

        if ret_res != 0:
            return ret_res

        elapsed_time = time.perf_counter() - start_time
        print('[ SUCCESS ] Total execution time: {:.2f} seconds. '.format(elapsed_time.total_seconds()))

        return ret_res