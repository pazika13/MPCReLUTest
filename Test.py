# import the libraries
from NssMPC.secure_model.mpc_party.semi_honest import SemiHonestCS
from NssMPC import ArithmeticSecretSharing
from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.crypto.aux_parameter.beaver_triples.arithmetic_triples import MatmulTriples
from NssMPC.config.runtime import PartyRuntime
import threading

import torch
DEVICE = "cuda"
# data belong to server
import threading

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


from NssMPC.config.configs import DEVICE

# data belong to server
x = RingTensor.convert_to_ring(torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=DEVICE))
# data belong to client
y = RingTensor.convert_to_ring(torch.tensor([[-1.0, 2.0], [4.0, 3.0]], device=DEVICE))

# split x into 2 parts
X = ArithmeticSecretSharing.share(x, 2)

# split y into 2 parts
Y = ArithmeticSecretSharing.share(y, 2)

temp_shared_x0=X[0]
temp_shared_x1=X[1]
temp_shared_y0=Y[0]
temp_shared_y1=Y[1]

def server_action(shared_x0):
    with PartyRuntime(server):
        two_ring = RingTensor.convert_to_ring(torch.tensor([[2.0, 2.0], [2.0, 2.0]]))
        res_0 = shared_x0 + two_ring
        #shared_x0.restore()
        restored_x = res_0.restore()
        print("x0:"+str(restored_x))
        real_x = restored_x.convert_to_real_field()
        print("\n x0 after restoring:", real_x)



def client_action(shared_x1):
    with PartyRuntime(client):
        ten_ring = RingTensor.convert_to_ring(torch.tensor([[10.0, 10.0], [10.0, 10.0]]))
        res_1 = shared_x1 + ten_ring
        #shared_x1.restore()
        restored_x  = res_1.restore()
        print("x1"+str(restored_x))
        real_x = restored_x.convert_to_real_field()
        print("\n x1 after restoring:", real_x)
if __name__ == "__main__":
    server_thread = threading.Thread(target=server_action, args=(temp_shared_x0,))
    client_thread = threading.Thread(target=client_action, args=(temp_shared_x1,))

    server_thread.start()
    client_thread.start()
    client_thread.join()
    server_thread.join()