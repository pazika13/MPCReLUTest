import time
import unittest
import torch
import crypten
import os
import sys
from multiprocess_launcher import MultiProcessLauncher
import crypten.nn as cnn
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
device = "cuda"

def CrtptenReLU():
    plaintext_input = torch.tensor([1,2,3,-2,-6]) # torch.randn(100, 5)  # 10x100 的张量
    #plaintext_input = torch.randn(512, 3072)
    plaintext_input = torch.randn(52, 307)
    start = time.time()
    crypten_input = crypten.cryptensor(plaintext_input,device=device)
    print("\n正在执行 ReLU...")
    crypten.reset_communication_stats()
    relu_output = crypten_input.relu()
    print(str(crypten.get_communication_stats()))
    temp = relu_output.get_plain_text()
    print(temp)
    spent = time.time() - start
    print("ReLU 执行完毕。用时:"+str(spent))

def CrtptenReLUMain():
    launcher = MultiProcessLauncher(2, CrtptenReLU)
    launcher.start()
    launcher.join()
    launcher.terminate()

# 运行测试
if __name__ == "__main__":
    CrtptenReLUMain()