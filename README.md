# py-aiger-discrete
Library for modeling functions over discrete sets using aiger circuits.

[![Build Status](https://cloud.drone.io/api/badges/mvcisback/py-aiger-discrete/status.svg)](https://cloud.drone.io/mvcisback/py-aiger-discrete)
[![docs badge](https://img.shields.io/badge/docs-docs-black)](https://mjvc.me/py-aiger-discrete)
[![codecov](https://codecov.io/gh/mvcisback/py-aiger-discrete/branch/main/graph/badge.svg)](https://codecov.io/gh/mvcisback/py-aiger-discrete)
[![PyPI version](https://badge.fury.io/py/py-aiger_discrete.svg)](https://badge.fury.io/py/py-aiger-discrete)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Installation

If you just need to use `py-aiger-discrete`, you can just run:

`$ pip install py-aiger-discrete`

For developers, note that this project uses the
[poetry](https://poetry.eustace.io/) python package/dependency
management tool. Please familarize yourself with it and then
run:

`$ poetry install`

   
# About

This library helps with modeling functions over finite sets using circuits. 

    f : A → B₁ × B₂ × … × Bₘ

where `A ⊆ A₁ × A₂ × … × Aₙ`.

This is done by providing:

1. A encoder/decoder pair for each set to and from (unsigned)
   integers. 
2. A circuit that uses the bit-vector representation of these integers
   to compute `f`.
3. A circuit that monitors if the input bit-vector sequence is
   `valid`, i.e., a member of `A`.

Functionally, the `py-aiger-discrete` library centers around the
`aiger_discrete.FiniteFunc` class which has 4 attributes an `aiger_bv.AIGBV` object.

1. A string `valid_id` indicating
1. A circuit, `circ`, over named bit-vectors in the form of an
   `aiger_bv.AIGBV` object. One of the outputs must be named
   `valid_id`.
1. A mapping from **inputs** to `aiger_discrete.Encoding` objects which
   encode/decode objects to integers. The standard bit-encoding of
   unsigned integers is feed into `circ`.
1. A mapping from **outputs** to `aiger_discrete.Encoding` objects which
   encode/decode objects to integers. These encodings are used to decode
   the resulting bit-vectors of `circ`.

# Usage

Below we provide a basic usage example. This example assumes basic
knowledge of the `py-aiger` ecosystem and particularly `py-aiger-bv`.

```python
import aiger_bv as BV

from aiger_discrete import Encoding, from_aigbv

# Will assume inputs are in 'A', 'B', 'C', 'D', or 'E'.
ascii_encoder = Encoding(
    decode=lambda x: chr(x + ord('A')),  # Make 'A' map to 0.
    encode=lambda x: ord(x) - ord('A'),
)

# Create function which maps: A -> B, B -> C, C -> D, D -> E.

x = BV.uatom(3, 'x')  # need 3 bits to capture 5 input types.
update_expr = (x + 1) & 0b111
circ = update_expr.with_output('y').aigbv

# Need to assert that the inputs are less than 4.
circ |= (x <= 4).with_output('##valid').aigbv

func = from_aigbv(
    circ,
    input_encodings={'x': ascii_encoder},
    output_encodings={'y': ascii_encoder},
    valid_id='##valid',
)

assert func('A') == 'B'
assert func('B') == 'C'
assert func('C') == 'D'
assert func('D') == 'E'
assert func('E') == 'A'
```

Note that `py-aiger-discrete` implements most of the circuit API as `aiger_bv.AIGBV`. 

For example, sequential composition:

```python
func12 = func1 >> func2
```

or parallel composition:

```python
func12 = func1 | func2
```

or unrolling:

```python
func_unrolled = func1.unroll(5)
```

or feedback: 


```python
func_cycle = func1.loopback({
    'input': 'x',
    'output': 'y',
    'keep_output': True,
    `input_encoder`: True,
    `init`: 'A',
})
```

Note that feedback now supports additional flag per wiring description
called `input_encoder` which determines if the input or output
encoding is used for initial latch value resp. The default is the
input encoding.


or renaming:
```python
func_renamed = func1['i', {'x': 'z'}]
assert func1.inputs == {'z'}
```
