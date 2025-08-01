import torch

from NssMPC import RingTensor, ArithmeticSecretSharing
from NssMPC.common.random.prg import PRG
from NssMPC.common.utils import convert_tensor
from NssMPC.config.configs import LAMBDA, DEVICE, PRG_TYPE
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.cw import CW
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dcf_key import DCFKey
from NssMPC.crypto.primitives import DCF


class FastSecNetReLUKey:
    def __init__(self,dcf_key,r_ss,b_ss):
        self.dcf_key = dcf_key
        self.r_ss = r_ss
        self.b_ss = b_ss

class FastSecNetReLU:

    @staticmethod
    def gen(num_of_keys, alpha,device="cpu"):
        #b = RingTensor.stack([RingTensor.ones_like(alpha),alpha],dim=0)
        b = RingTensor.convert_to_ring(torch.Tensor([1, -1 * alpha.convert_to_real_field()[0].item()],device=device))
        dcf_key0,dcf_key1 =  DCFKey.gen(num_of_keys, alpha, -1 * b)
        R = ArithmeticSecretSharing.share(alpha)
        B = ArithmeticSecretSharing.share(b)
        return FastSecNetReLUKey(dcf_key0,R[0],B[0]),FastSecNetReLUKey(dcf_key1,R[1],B[1])

    @staticmethod
    def eval(x_ss:ArithmeticSecretSharing, key:FastSecNetReLUKey, party_id):
        k_ss = key.r_ss + x_ss
        x_r = k_ss.restore()
        res = DCF.eval(x_r,key.dcf_key,party_id) + key.b_ss
        #print(x_r.convert_to_real_field())
        y = res[...,0] * x_r + res[...,1]
        return y
