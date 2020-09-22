import pytest

import aiger_bv as BV

from aiger_discrete import DiscreteCirc, Encoding, from_aigbv


def test_discrete_wrapper_smoke():
    x = BV.uatom(3, 'x')
    circ = (x + 1).with_output('z') \
                  .aigbv

    func = from_aigbv(circ)
    assert func({'x': (0, 1, 1)})[0] == {'z': (1, 1, 1)}

    int_enc = Encoding(
        encode=lambda x: BV.encode_int(3, x, signed=False),
        decode=lambda x: BV.decode_int(x, signed=False),
    )

    func2 = from_aigbv(
        circ,
        input_encodings={'x': int_enc},
        output_encodings={'z': int_enc},
    )        
    assert func2({'x': 2})[0] == {'z': 3}

    valid = (x <= 2).with_output('##valid') \
                    .aigbv

    func3 = from_aigbv(
        circ | valid,
        input_encodings={'x': int_enc},
        output_encodings={'z': int_enc},
    )
    assert func3({'x': 2})[0] == {'z': 3}    

    with pytest.raises(ValueError):
        func3({'x': 3})
