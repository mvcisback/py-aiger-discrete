import aiger_bv as BV
from bidict import bidict

from aiger_discrete import Encoding, from_aigbv
from aiger_discrete.mdd import to_mdd


ONE_HOT = bidict({
    0b000: 0b00001,
    0b001: 0b00010,
    0b010: 0b00100,
    0b011: 0b01000,
    0b100: 0b10000,
})


def test_readme_mdd():
    # Will assume inputs are in 'A', 'B', 'C', 'D', or 'E'.
    ascii_encoder = Encoding(
        decode=lambda x: chr(x + ord('A')),  # Make 'A' map to 0.
        encode=lambda x: ord(x) - ord('A'),
    )

    one_hot_ascii_encoder = Encoding(
        decode=lambda x: ascii_encoder.decode(ONE_HOT.inv[x]),
        encode=lambda x: ONE_HOT[ascii_encoder.encode(x)],
    )

    # Create function which maps: A -> B, B -> C, C -> D, D -> E.
    x = BV.uatom(3, 'x')  # Need 3 bits to capture 5 input types.
    update_expr = (x < 4).repeat(3) & (x + 1)  # 0 if x < 4 else x + 1.
    circ = update_expr.with_output('y').aigbv
    circ |= (x < 5).with_output('##valid').aigbv
    one_hot_converter = BV.lookup(
        3, 5, ONE_HOT, 'y', 'y',
        in_signed=False, out_signed=False
    )
    circ >>= one_hot_converter

    func_circ = from_aigbv(
        circ,
        input_encodings={'x': ascii_encoder},
        output_encodings={'y': one_hot_ascii_encoder},
        valid_id='##valid',
    )

    assert func_circ({'x': 'A'})[0] == {'y': 'B'}
    assert func_circ({'x': 'B'})[0] == {'y': 'C'}
    assert func_circ({'x': 'C'})[0] == {'y': 'D'}
    assert func_circ({'x': 'D'})[0] == {'y': 'E'}
    assert func_circ({'x': 'E'})[0] == {'y': 'A'}

    func_mdd = to_mdd(func_circ)

    assert func_mdd({'x': 'A'})[0] == 'B'
    assert func_mdd({'x': 'B'})[0] == 'C'
    assert func_mdd({'x': 'C'})[0] == 'D'
    assert func_mdd({'x': 'D'})[0] == 'E'
    assert func_mdd({'x': 'E'})[0] == 'A'
