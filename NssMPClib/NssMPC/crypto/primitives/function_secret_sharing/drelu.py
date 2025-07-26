from NssMPC import RingTensor
from NssMPC.config import HALF_RING, BIT_LEN
from NssMPC.crypto.aux_parameter import SigmaDICFKey
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.drelu_key import UnauthSharkDReLUKey
from NssMPC.crypto.primitives import DCF
from NssMPC.crypto.primitives.boolean_secret_sharing import BooleanSecretSharing
from NssMPC.crypto.primitives.function_secret_sharing.dpf import prefix_parity_query


class UnauthSharkDReLU:

    @staticmethod
    def gen(x_0,bit_len = BIT_LEN):
        return UnauthSharkDReLUKey.gen(x_0, bit_len)

    @staticmethod
    def eval(x_shift: RingTensor, key, party_id):

        to_minus = 1 << (x_shift.bit_len - 1)
        c = DCF.eval(x_shift - to_minus, key.dcf_key, party_id)
        return BooleanSecretSharing(c ^ x_shift.signbit() ^ key.one_bss)

    @staticmethod
    def one_key_eval(input_list, key, party_id):
        num = len(input_list)
        x_shift = RingTensor.stack(input_list)
        shape = x_shift.shape
        x_shift = x_shift.view(num, -1, 1)
        y = x_shift % (HALF_RING - 1)
        y = y + 1
        out = prefix_parity_query(y, key.dpf_key, party_id)
        out = x_shift.signbit() * party_id ^ key.c.view(-1, 1) ^ out
        return out.view(shape)
