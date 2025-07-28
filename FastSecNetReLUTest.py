import os
import random

import torch
import time
import threading

from FastSecNet import FastSecNetReLUKey, FastSecNetReLU
from NssMPC import ArithmeticSecretSharing, RingTensor
from NssMPC.secure_model.mpc_party.semi_honest import SemiHonestCS
from NssMPC.config.runtime import PartyRuntime
from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICF
from NssMPC.crypto.aux_parameter import SigmaDICFKey

# 1. 初始化两方计算环境 (服务器和客户端)
# 这个设置模仿了 Tutorial_2 中的多线程模拟方法
server = SemiHonestCS(type='server')
client = SemiHonestCS(type='client')


def setup_party(party):
    """一个通用函数来启动和连接一个计算方"""
    with PartyRuntime(party):
        party.online()


# 在不同的线程中启动服务器和客户端
server_thread = threading.Thread(target=setup_party, args=(server,))
client_thread = threading.Thread(target=setup_party, args=(client,))
server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# 2. 定义 ReLU 计算函数
def fastsecnet_relu_server(x_shift,key0):
    """服务器端的计算逻辑"""
    with PartyRuntime(server):
        start = time.time()
        res_0 = FastSecNetReLU.eval(x_ss=x_shift, key=key0, party_id=0)
        spent_time = time.time() - start
        print("time for get secret share:" + str(spent_time))
        #res_1 = server.receive()

        final_result = res_0.restore().convert_to_real_field() #+ res_1
        print("final:"+str(final_result))
        spent_time = time.time() - start
        print("time for get plaintext:"+str(spent_time))

def fastsecnet_relu_client(x_shift,key1):
    with PartyRuntime(client):
        res_1 = FastSecNetReLU.eval(x_ss=x_shift, key=key1, party_id=1)
        #client.send(res_1)
        res_1.restore()

if __name__ == "__main__":
    plaintext_input = torch.tensor([[30000000.,0.2,3.,-0.4,-5.,1.,-6.4,1.],[1.,2.,3.,-0.4,-5.,1.,-6.5,1.]])
    plaintext_input = torch.randn(52, 307)
    #plaintext_input = torch.randn(10, 5)g
    #plaintext_input = torch.randn(12, 3072)
    #plaintext_input = torch.tensor([[1,2,3,-4,-5,1,-6,1],[1,2,3,-4,-5,1,-6,1]])
    num_elements = plaintext_input.numel()

    # 将 torch.tensor 转换为 RingTensor
    x_ring = RingTensor.convert_to_ring(plaintext_input)
    X = ArithmeticSecretSharing.share(x_ring, 2)
    #r = RingTensor.random([1],down_bound=-3000,upper_bound=3000).to("float")
    r = RingTensor.convert_to_ring(torch.tensor([100000.]))
    #r = RingTensor.convert_to_ring(torch.tensor([3000000000.]))
    key0, key1 = FastSecNetReLU.gen(num_of_keys=num_elements,alpha=r)
    print(plaintext_input)
    server_relu_thread = threading.Thread(target=fastsecnet_relu_server, args=(X[0],key0))
    client_relu_thread = threading.Thread(target=fastsecnet_relu_client, args=(X[1],key1))

    server_relu_thread.start()
    client_relu_thread.start()
    client_relu_thread.join()
    server_relu_thread.join()


    server.close()
    client.close()