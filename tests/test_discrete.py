import pytest

import aiger_bv as BV

from aiger_discrete import Encoding, from_aigbv


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

    assert func3.inputs == {'x'}
    assert func3.outputs == {'z'}
    assert func3.latches == set()
    assert func3.latch2init == {}


def test_rename_valid():
    func = from_aigbv(BV.uatom(3, 'x').aigbv).rename_valid('foo')
    assert 'foo' in func.circ.outputs
    assert 'foo' == func.valid_id


def test_parallel_composition():
    x = BV.uatom(3, 'x')
    circ1 = (x + 1).with_output('y').aigbv \
        | (x < 5).with_output('##valid').aigbv
    func1 = from_aigbv(circ1)

    circ2 = x.with_output('z').aigbv \
        | (x > 2).with_output('##valid').aigbv
    func2 = from_aigbv(circ2)

    func12 = func1 | func2
    assert func12({'x': (1, 1, 0)})[0] == {
        'y': (0, 0, 1), 'z': (1, 1, 0)
    }

    with pytest.raises(ValueError):
        func12({'x': (0, 0, 0)})

    with pytest.raises(ValueError):
        func12({'x': (1, 1, 1)})
