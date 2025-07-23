import torch
import time
import threading

from NssMPC import ArithmeticSecretSharing
from NssMPC.crypto.aux_parameter.select_keys import SelectLinKey
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import b2a
from NssMPC.crypto.protocols.selection.selectlin import SelectLin
from NssMPC.secure_model.mpc_party.semi_honest import SemiHonestCS
from NssMPC.config.runtime import PartyRuntime
from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.config import SCALE_BIT, GELU_TABLE_BIT
from NssMPC.crypto.aux_parameter.look_up_table_keys.gelu_key import GeLUKey
from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICF
from NssMPC.crypto.aux_parameter import SigmaDICFKey, B2AKey
from NssMPC.secure_model.utils.param_provider import ParamProvider

# 1. 初始化两方计算环境 (服务器和客户端)
# 这个设置模仿了 Tutorial_2 中的多线程模拟方法
server = SemiHonestCS(type='server')
client = SemiHonestCS(type='client')


def setup_party(party):
    """一个通用函数来启动和连接一个计算方"""
    with PartyRuntime(party):
        party.append_provider(ParamProvider(param_type=SelectLinKey))
        party.append_provider(ParamProvider(param_type=B2AKey))
        party.online()


# 在不同的线程中启动服务器和客户端
server_thread = threading.Thread(target=setup_party, args=(server,))
client_thread = threading.Thread(target=setup_party, args=(client,))
server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# 2. 定义 ReLU 计算函数
def sigma_relu_server(x_ss,x_shift,dicf_key0,select_lin_key0):
    """服务器端的计算逻辑"""
    with PartyRuntime(server):
        start = time.time()
        #select_lin_key0 = PartyRuntime.party.get_param(SelectLinKey, x_shift.numel())
        drelu_bss = SigmaDICF.eval(x_shift=x_shift, key=dicf_key0, party_id=0)
        spent_time = time.time() - start
        drelu_ass = b2a(drelu_bss,PartyRuntime.party)
        relu_ss_0 = SelectLin.eval(x_ss.flatten(), drelu_ass.flatten(), RingTensor.zeros_like(x_shift.flatten()), select_lin_key0)
        # drelu_res_1 = server.receive()
        # final_res = drelu_res_1 ^ drelu_res_0
        #final_res = relu_res_0.restore()
        #final_result = res_0 ^ res_1
        #print(final_res)
        #relu_res = relu_ss_0.restore().convert_to_real_field()
        spent_time = time.time() - start
        print("time for get plaintext:"+str(spent_time))

def sigma_relu_client(x_ss,x_shift,dicf_key1,select_lin_key1):
    with PartyRuntime(client):
        drelu_bss = SigmaDICF.eval(x_shift=x_shift, key=dicf_key1, party_id=1)
        drelu_ass = b2a(drelu_bss, PartyRuntime.party)
        relu_ss_0 = SelectLin.eval(x_ss.flatten(), drelu_ass.flatten(), RingTensor.zeros_like(x_shift.flatten()), select_lin_key1)
        client.send(drelu_ass)



if __name__ == "__main__":
    plaintext_input = torch.tensor([[1,2,3,-4,-5,1,-6],[1,2,3,-4,-5,1,-6]]) #torch.randn(100, 5)
    #plaintext_input = torch.randn(512, 3072)
    num_elements = plaintext_input.numel()
    x_ring = RingTensor.convert_to_ring(plaintext_input)
    p = RingTensor([0,0,1,0]).repeat(num_elements,1)
    q = RingTensor([0,0,0,0]).repeat(num_elements,1)
    SelectLinKey.gen_and_save(num_elements,None, p, q)
    B2AKey.gen_and_save(plaintext_input.numel())
    select_lin_key0, select_lin_key1 = SelectLinKey.gen(plaintext_input.numel(),p, q)
    # 将 torch.tensor 转换为 RingTensor

    X = ArithmeticSecretSharing.share(x_ring, 2)
    dicf_key0, dicf_key1 = SigmaDICFKey.gen(num_of_keys=num_elements)

    x_shift = dicf_key0.r_in.reshape(x_ring.shape) + dicf_key1.r_in.reshape(x_ring.shape) + x_ring

    server_relu_thread = threading.Thread(target=sigma_relu_server, args=(X[0],x_shift,dicf_key0,select_lin_key0))
    client_relu_thread = threading.Thread(target=sigma_relu_client, args=(X[1],x_shift,dicf_key1,select_lin_key1))

    server_relu_thread.start()
    client_relu_thread.start()
    client_relu_thread.join()
    server_relu_thread.join()


    server.close()
    client.close()