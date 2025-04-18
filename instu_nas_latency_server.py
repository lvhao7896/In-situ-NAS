import socket
import os
import struct
import ast
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import threading
import hashlib
import psutil
import time

target_platform = 'VPU'
if target_platform.upper() == 'VPU':
    from estimators.VPU_Estimator import VPU_Estimator as Estimator
elif target_platform.upper() == 'TPU':
    from estimators.TPU_Estimator import TPU_Estimator as Estimator
elif target_platform.upper() == 'DPU':
    from estimators.DPU_Estimator import DPU_Estimator as Estimator
elif target_platform.upper() == 'ARM':
    # from estimators.ARM_Estimator import ARM_Estimator as Estimator
    class Estimator():
        def __init__(self, model_dir='./model_pool/'):
            self.model_dir = model_dir
            process = os.popen('adb push ./benchmark_model /data/local/tmp/ && adb shell chmod +x /sdcard/benchmark_model')
            process_out = process.read()
            process.close()

        def latency_eval(self, subnet_name):
            os.system('adb push ./{}/{}.tflite /data/local/tmp'.format(self.model_dir, subnet_name))
            process = os.popen('adb shell /data/local/tmp/benchmark_model \
                                    --graph=/data/local/tmp/{}.tflite \
                                    --num_threads=1  --use_gpu=false --use_xnnpack=false --verbose'.format(subnet_name))
            process_out = process.read()
            process.close()
            for idx, line in enumerate(process_out.splitlines()):
                # print(idx, line)
                pattern_str = 'Inference (avg):'
                pidx = line.find(pattern_str)
                # print(process_out)
                if pidx > 0:
                    act_efficiency = float(line[pidx+len(pattern_str):])/1000.
                    print(act_efficiency)
            return (act_efficiency, 0)
elif target_platform.upper() == 'CUSTOM':
    from estimators.Custom_Estimator import Custom_Estimator as Estimator
else:
    raise NotImplemented

def server_config():
    p = psutil.Process()
    cpu_list = [1]
    p.cpu_affinity(cpu_list)

def start_eval_service(recv_eval_queue, save_dir):
    print("eval deamon service started")
    eval_count = 0
    estimator = Estimator(save_dir)
    latency = 1e5
    std = 0
    while True:
        subnet_name, client_socket = recv_eval_queue.get()
        print("eval ", subnet_name)
        try:
            eval_start = time.time()
            latency, std = estimator.latency_eval(subnet_name)
            # print(latency)
            eval_count += 1
            if eval_count % 1000 == 0:
                estimator.clean()
                # VPU estimator
                estimator = Estimator(save_dir)
            eval_end = time.time()
            print("eval time : ", eval_end-eval_start)
        except Exception as e:
            print("Eval Error ", e)
        try :
            send_start = time.time()
            result_format = 'ff'
            send_data = struct.pack(result_format, latency, std)
            client_socket.sendall(send_data)
            client_socket.close()
            send_end = time.time()
            print("send time : ", send_end-send_start)
        except Exception as e:
            print("Msg send back Error ", e)

def cal_md5(file_path):
    with open(file_path, 'rb') as fr:
        md5 = hashlib.md5()
        md5.update(fr.read())
        md5 = md5.hexdigest()
        return md5

def recv_model(socket, recv_eval_queue, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    eval_cmd_format = '128sII'
    file_info_format = '128sIQ32s'
    max_msg_len = 1024
    try:
        recv_data = socket.recv(struct.calcsize(eval_cmd_format))
        subnet_name, subnet_name_len, file_num = struct.unpack(eval_cmd_format, recv_data)
        subnet_name = subnet_name.decode('utf-8')[:subnet_name_len]
        for _ in range(file_num):
            file_info = socket.recv(struct.calcsize(file_info_format))
            file_name, file_name_len, file_size, md5 = struct.unpack(file_info_format, file_info)
            file_name = file_name.decode('utf-8')[:file_name_len]
            md5 = md5.decode('utf-8')
            recv_sz = 0
            save_path = save_dir + file_name
            
            with open(save_path, 'wb') as f:
                while recv_sz < file_size:
                    if file_size - recv_sz > max_msg_len:
                        recv_data = socket.recv(max_msg_len)
                        recv_sz += len(recv_data)
                    else:
                        recv_data = socket.recv(file_size - recv_sz)
                        recv_sz += len(recv_data)
                    f.write(recv_data)
            # print('file size ', file_size, ' recieved size ', recv_sz)
            recv_md5 = cal_md5(save_path)
            # print("recv file ", file_name)
            if md5 != recv_md5:
                raise Exception("MD5 check failed ", md5, " vs. ", recv_md5)  
        recv_eval_queue.put_nowait((subnet_name, socket))
    except Exception as e:
        print("Recv Error ! ", e)

if __name__ == '__main__':
    port = 20027
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_server_socket.bind(("", port))
    print("Listen on ", port)
    tcp_server_socket.listen(1024)
    max_msg_len = 1024
    encoding = 'utf-8'
    save_dir = './recv_model_dir/'
    server_config()
    # start eval service
    recv_eval_queue = Queue()
    eval_deamon_thread = threading.Thread(target=start_eval_service, args=(recv_eval_queue, save_dir))
    eval_deamon_thread.start()
    idx = 0
    with ThreadPoolExecutor() as executor:
        while True:
            client_socket, client_ip = tcp_server_socket.accept()
            print("Client IP : ", client_ip)
            print("Accept conn idx : {}", idx)
            idx += 1
            executor.submit(recv_model, client_socket, recv_eval_queue, save_dir)