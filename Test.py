# import the libraries
from NssMPC.crypto.primitives.function_secret_sharing.dcf import DCF
from NssMPC.crypto.aux_parameter import DCFKey
from NssMPC import RingTensor
if __name__ == "__main__":
    # import the libraries
    from NssMPC.crypto.primitives.function_secret_sharing.dcf import DCF
    from NssMPC.crypto.aux_parameter import DCFKey
    from NssMPC import RingTensor

    num_of_keys = 10  # We need a few keys for a few function values, but of course we can generate many keys in advance.

    # generate keys in offline phase
    # set alpha and beta
    alpha = RingTensor.convert_to_ring(5)
    beta = RingTensor.convert_to_ring(1)

    Key0, Key1 = DCF.gen(num_of_keys=num_of_keys, alpha=alpha, beta=beta)

    # online phase
    # generate some values what we need to evaluate
    x = RingTensor.convert_to_ring([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Party 0:
    res_0 = DCF.eval(x=x, keys=Key0, party_id=0)

    # Party 1:
    res_1 = DCF.eval(x=x, keys=Key1, party_id=1)

    # restore result
    res = res_0 + res_1
    print(res)