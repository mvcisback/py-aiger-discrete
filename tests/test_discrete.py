import pytest

import aiger_bv as BV

from aiger_discrete import Encoding, from_aigbv


NEG_ENC = Encoding(
    encode=lambda x: -x,
    decode=lambda x: -x,
)
INT_ENC = Encoding()


def test_discrete_wrapper_smoke():
    x = BV.uatom(3, 'x')
    circ = (x + 1).with_output('z') \
                  .aigbv

    func = from_aigbv(circ)
    assert func({'x': 6})[0] == {'z': 7}

    func2 = from_aigbv(
        circ,
        input_encodings={'x': NEG_ENC},
        output_encodings={'z': NEG_ENC},
    )
    assert func2({'x': -2})[0] == {'z': -3}

    valid = (x <= 2).with_output('##valid') \
                    .aigbv

    func3 = from_aigbv(
        circ | valid,
        input_encodings={'x': NEG_ENC},
        output_encodings={'z': NEG_ENC},
    )
    assert func3({'x': -2})[0] == {'z': -3}

    with pytest.raises(ValueError):
        func3({'x': -3})

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
    func1 = from_aigbv(
        circ1,
        input_encodings={'x': INT_ENC},
        output_encodings={'y': INT_ENC},
    )

    circ2 = x.with_output('z').aigbv \
        | (x > 2).with_output('##valid').aigbv
    func2 = from_aigbv(
        circ2,
        input_encodings={'x': INT_ENC},
        output_encodings={'z': INT_ENC},
    )

    func12 = func1 | func2
    assert func12({'x': 3})[0] == {'y': 4, 'z': 3}

    with pytest.raises(ValueError):
        func12({'x': 0})

    with pytest.raises(ValueError):
        func12({'x': 7})


def test_seq_composition():
    x = BV.atom(4, 'x')
    y = BV.atom(4, 'y')

    circ1 = (x + 1).with_output('y').aigbv \
        | (x < 5).with_output('##valid').aigbv
    func1 = from_aigbv(circ1,)

    circ2 = (y - 1).with_output('z').aigbv \
        | (y > 2).with_output('##valid').aigbv
    func2 = from_aigbv(circ2,)

    func12 = func1 >> func2
    assert func12({'x': 4})[0] == {'z': 4}

    with pytest.raises(ValueError):
        func12({'x': 1})

    func12 = func2 << func1
    assert func12({'x': 4})[0] == {'z': 4}

    with pytest.raises(ValueError):
        func12({'x': 1})


def test_relabel():
    x = BV.uatom(3, 'x')
    circ1 = (x + 1).with_output('y').aigbv \
        | (x < 5).with_output('##valid').aigbv
    func1 = from_aigbv(circ1,)
    assert func1['i', {'x': 'z'}].inputs == {'z'}
    assert func1['o', {'y': 'z'}].outputs == {'z'}
    assert func1['i', {'x': 'z'}].valid_id == func1.valid_id


def test_loopback_and_unroll():
    x = BV.uatom(3, 'x')
    y = BV.uatom(3, 'y')
    circ1 = (x + y).with_output('y').aigbv \
        | (x < 7).with_output('##valid').aigbv
    func1 = from_aigbv(circ1,)
    func2 = func1.loopback({
        'input': 'x', 'output': 'y',
        'keep_output': True,
        'init': 0,
    })
    assert func2.simulate([{'y': 1}, {'y': 1}])[-1][0] == {'y': 2}
    with pytest.raises(ValueError):
        assert func2.simulate([{'y': 1}]*10)

    func3 = func2.unroll(2, only_last_outputs=True)
    assert func3({'y##time_0': 0, 'y##time_1': 1})[0] == {'y##time_2': 1}

    with pytest.raises(ValueError):
        assert func3({'y##time_0': 7, 'y##time_1': 0})


def test_readme():
    # Will assume inputs are in 'A', 'B', 'C', 'D', or 'E'.
    ascii_encoder = Encoding(
        decode=lambda x: chr(x + ord('A')),  # Make 'A' map to 0.
        encode=lambda x: ord(x) - ord('A'),
    )

    # Create function which maps: A -> B, B -> C, C -> D, D -> E.
    x = BV.uatom(3, 'x')  # Need 3 bits to capture 5 input types.
    update_expr = (x < 4).repeat(3) & (x + 1)  # 0 if x < 4 else x + 1.
    circ = update_expr.with_output('y').aigbv

    # Need to assert that the inputs are less than 4.
    circ |= (x < 5).with_output('##valid').aigbv

    # Wrap using aiger_discrete.
    func = from_aigbv(
        circ,
        input_encodings={'x': ascii_encoder},
        output_encodings={'y': ascii_encoder},
        valid_id='##valid',
    )

    assert func({'x': 'A'})[0] == {'y': 'B'}
    assert func({'x': 'B'})[0] == {'y': 'C'}
    assert func({'x': 'C'})[0] == {'y': 'D'}
    assert func({'x': 'D'})[0] == {'y': 'E'}
    assert func({'x': 'E'})[0] == {'y': 'A'}
