import torch
import time
import threading
from NssMPC.crypto.aux_parameter import *
from NssMPC import ArithmeticSecretSharing
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import b2a
from NssMPC.crypto.protocols.look_up_table import LookUp
from NssMPC.crypto.protocols.selection.selectlin import SelectLin
from NssMPC.secure_model.mpc_party.semi_honest import SemiHonestCS
from NssMPC.config.runtime import PartyRuntime
from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.config import SCALE_BIT, GELU_TABLE_BIT
from NssMPC.crypto.aux_parameter.look_up_table_keys.gelu_key import GeLUKey
from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICF
from NssMPC.crypto.aux_parameter import SigmaDICFKey
from NssMPC.secure_model.utils.param_provider import ParamProvider

# 1. 初始化两方计算环境 (服务器和客户端)
# 这个设置模仿了 Tutorial_2 中的多线程模拟方法
server = SemiHonestCS(type='server')
client = SemiHonestCS(type='client')


def setup_party(party):
    """一个通用函数来启动和连接一个计算方"""
    with PartyRuntime(party):
        party.append_provider(ParamProvider(param_type=GeLUKey))
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
def sigma_relu_server(x):
    """服务器端的计算逻辑"""
    with PartyRuntime(server):

        table_scale_bit = GELU_TABLE_BIT
        shape = x.shape
        x = x.flatten()

        gelu_key = PartyRuntime.party.get_param(GeLUKey, x.numel())
        sigma_key = gelu_key.sigma_key
        select_lin_key = gelu_key.select_lin_key
        select_key = gelu_key.select_key

        x_r_in = gelu_key.sigma_key.r_in
        x_shift = ArithmeticSecretSharing(x_r_in) + x.flatten()
        x_shift = x_shift.restore()

        y_shift = x_shift // (x.scale // (2 ** table_scale_bit))
        y_shift.bit_len = x.bit_len - SCALE_BIT + table_scale_bit

        d_and_w = SigmaDICF.one_key_eval(
            [y_shift, y_shift + (2 ** (table_scale_bit + 2) - 1), y_shift - (2 ** (table_scale_bit + 2))], sigma_key,
            PartyRuntime.party.party_id)
        d = d_and_w[0]
        w = d_and_w[1] ^ d_and_w[2]

        d_and_w_b = RingTensor.cat([d, w], dim=0)
        d_and_w_a = b2a(d_and_w_b, PartyRuntime.party)
        d = d_and_w_a[:d.numel()]
        w = d_and_w_a[d.numel():]

        w_shift = ArithmeticSecretSharing(select_lin_key.w) + w.flatten()
        d_shift = ArithmeticSecretSharing(select_lin_key.d) + d.flatten()

        length = w_shift.numel()
        w_and_d = ArithmeticSecretSharing.cat([w_shift, d_shift], dim=0).restore()
        w_shift = w_and_d[:length]
        d_shift = w_and_d[length:]

        c = SelectLin.eval(y_shift, w_shift, d_shift, select_lin_key)

        s_shift = d_shift % 2
        s_shift.bit_len = d_shift.bit_len
        relu_x = _gelu_select_eval(x_shift, s_shift, select_key, select_lin_key.d, x_r_in, PartyRuntime.party)
        relu_x.dtype = x.dtype
        relu_x = relu_x.reshape(shape)
        #res = relu_x - LookUp.eval(c, gelu_key.look_up_key, gelu_key.look_up_table).reshape(shape)
        final_relu_x =  relu_x.restore()
        #temp = res.restore()
        print(final_relu_x.convert_to_real_field())
        #print(temp.convert_to_real_field())

def sigma_relu_client(x):
    with PartyRuntime(client):
        table_scale_bit = GELU_TABLE_BIT
        shape = x.shape
        x = x.flatten()

        gelu_key = PartyRuntime.party.get_param(GeLUKey, x.numel())
        sigma_key = gelu_key.sigma_key
        select_lin_key = gelu_key.select_lin_key
        select_key = gelu_key.select_key

        x_r_in = gelu_key.sigma_key.r_in
        x_shift = ArithmeticSecretSharing(x_r_in) + x.flatten()
        x_shift = x_shift.restore()

        y_shift = x_shift // (x.scale // (2 ** table_scale_bit))
        y_shift.bit_len = x.bit_len - SCALE_BIT + table_scale_bit

        d_and_w = SigmaDICF.one_key_eval(
            [y_shift, y_shift + (2 ** (table_scale_bit + 2) - 1), y_shift - (2 ** (table_scale_bit + 2))], sigma_key,
            PartyRuntime.party.party_id)
        d = d_and_w[0]
        w = d_and_w[1] ^ d_and_w[2]

        d_and_w_b = RingTensor.cat([d, w], dim=0)
        d_and_w_a = b2a(d_and_w_b, PartyRuntime.party)
        d = d_and_w_a[:d.numel()]
        w = d_and_w_a[d.numel():]

        w_shift = ArithmeticSecretSharing(select_lin_key.w) + w.flatten()
        d_shift = ArithmeticSecretSharing(select_lin_key.d) + d.flatten()

        length = w_shift.numel()
        w_and_d = ArithmeticSecretSharing.cat([w_shift, d_shift], dim=0).restore()
        w_shift = w_and_d[:length]
        d_shift = w_and_d[length:]

        c = SelectLin.eval(y_shift, w_shift, d_shift, select_lin_key)

        s_shift = d_shift % 2
        s_shift.bit_len = d_shift.bit_len
        relu_x = _gelu_select_eval(x_shift, s_shift, select_key, select_lin_key.d, x_r_in, PartyRuntime.party)
        relu_x.dtype = x.dtype
        relu_x = relu_x.reshape(shape)
        relu_x.restore()
        #res = relu_x - LookUp.eval(c, gelu_key.look_up_key, gelu_key.look_up_table).reshape(shape)
        #res.restore()


def _gelu_select_eval(x_shift: RingTensor, s_shift, key, r_in_1, r_in_2, party):
    shape = x_shift.shape
    x_shift = x_shift.flatten()
    return ArithmeticSecretSharing(RingTensor.where(s_shift, (party.party_id - r_in_1) * x_shift - r_in_2 + key.w
                                                    , r_in_1 * x_shift + key.w - key.z).reshape(shape))
if __name__ == "__main__":

    plaintext_input = torch.tensor([[1,2,3,-4,-5,1,-6],[1,2,3,-4,-5,1,-6]]) #torch.randn(100, 5)
    GeLUKey.gen_and_save(plaintext_input.numel())
    B2AKey.gen_and_save(plaintext_input.numel())
    #plaintext_input = torch.randn(512, 3072)
    #num_elements = plaintext_input.numel()

    # 将 torch.tensor 转换为 RingTensor
    x_ring = RingTensor.convert_to_ring(plaintext_input)
    #x_ring.convert_to_real_field()
    X = ArithmeticSecretSharing.share(x_ring, 2)
    # key0, key1 = SigmaDICFKey.gen(num_of_keys=num_elements)
    #
    # x_shift = key0.r_in.reshape(x_ring.shape) + key1.r_in.reshape(x_ring.shape) + x_ring

    server_relu_thread = threading.Thread(target=sigma_relu_server, args=(X[0],))
    client_relu_thread = threading.Thread(target=sigma_relu_client, args=(X[1],))

    server_relu_thread.start()
    client_relu_thread.start()
    client_relu_thread.join()
    server_relu_thread.join()


    server.close()
    client.close()