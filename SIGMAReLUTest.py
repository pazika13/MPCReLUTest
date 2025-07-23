import torch
import threading

from NssMPC.application.neural_network.layers import SecGELU
from NssMPC.application.neural_network.layers.activation import SecReLUTest, SecReLU
from NssMPC.crypto.aux_parameter import GeLUKey, B2AKey, SigmaDICFKey, AssMulTriples
from NssMPC import ArithmeticSecretSharing
from NssMPC.secure_model.mpc_party.semi_honest import SemiHonestCS
from NssMPC.config.runtime import PartyRuntime
from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.secure_model.utils.param_provider import ParamProvider

# --- Main execution block ---
if __name__ == "__main__":
    # 1. 初始化两方计算环境
    server = SemiHonestCS(type='server')
    client = SemiHonestCS(type='client')

    def setup_party(party):
        with PartyRuntime(party):
            # 确保Provider被正确添加
            party.append_provider(ParamProvider(param_type=GeLUKey))
            party.append_provider(ParamProvider(param_type=B2AKey))
            party.append_provider(ParamProvider(param_type=SigmaDICFKey))
            party.append_provider(ParamProvider(param_type=AssMulTriples))
            party.online()

    server_thread = threading.Thread(target=setup_party, args=(server,))
    client_thread = threading.Thread(target=setup_party, args=(client,))
    server_thread.start()
    client_thread.start()
    client_thread.join()
    server_thread.join()
    print("--- Parties connected and ready ---")

    # 2. 准备数据和密钥
    #plaintext_input = torch.tensor([[-2.5, -1.0, 0.0, 0.5, 1.0, 2.5]], dtype=torch.float32)
    #plaintext_input = torch.tensor([[30000000.,0.2,3.,-0.4,-5.,1.,-6.4,1.],[1.,2.,3.,-0.4,-5.,1.,-6.5,1.]])
    #plaintext_input = torch.randn(12, 72)
    plaintext_input = torch.randn(512, 3072)
    print("Plaintext Input:\n", plaintext_input)

    # PyTorch的参考结果
    # torch_gelu = torch.nn.GELU()
    # plaintext_result = torch_gelu(plaintext_input).to(torch.float64).to("cuda")
    # print("Expected PyTorch Result:\n", plaintext_result)

    # 生成协议所需的密钥
    GeLUKey.gen_and_save(plaintext_input.numel())
    B2AKey.gen_and_save(plaintext_input.numel())
    SigmaDICFKey.gen_and_save(plaintext_input.numel())
    AssMulTriples.gen_and_save(plaintext_input.numel())

    # 将明文转换为RingTensor并进行秘密分享
    x_ring = RingTensor.convert_to_ring(plaintext_input)
    x_shared = ArithmeticSecretSharing.share(x_ring, 2)
    x_server, x_client = x_shared[0], x_shared[1]

    # 3. 定义各方的计算任务
    result_shares = {}

    def run_server_task(x_s):
        with PartyRuntime(server):
            # 直接实例化并调用SecGELU
            gelu_op = SecReLUTest()
            result_shares['server'] = gelu_op(x_s)
            # 5. 重构结果并验证
            final_result_shared = result_shares['server']
            # final_result_ring = final_result_shared.restore()
            # final_result_float = final_result_ring.convert_to_real_field()

            # print("Reconstructed MPC Result:\n", final_result_float)

            # 比较结果
            # 使用 atol (absolute tolerance) 来处理浮点数精度问题
            #assert torch.allclose(plaintext_result, final_result_float, atol=1e-3), "Test Failed: Results do not match!"
            #print("\n✅ Test Passed: MPC GeLU result matches PyTorch result!")


    def run_client_task(x_c):
        with PartyRuntime(client):
            gelu_op = SecReLUTest()
            result_shares['client'] = gelu_op(x_c)
            # result_shares['client'].restore()

    # 4. 在不同线程中执行安全计算
    server_task_thread = threading.Thread(target=run_server_task, args=(x_server,))
    client_task_thread = threading.Thread(target=run_client_task, args=(x_client,))

    server_task_thread.start()
    client_task_thread.start()
    server_task_thread.join()
    client_task_thread.join()
    print("--- Secure computation finished ---")


    # 6. 关闭连接
    server.close()
    client.close()