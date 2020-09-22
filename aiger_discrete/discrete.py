from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import aiger_bv as BV
import attr
import funcy as fn


# TODO: implement py-aiger API
@attr.s(auto_attribs=True, frozen=True)
class Encoding:
    encode: Callable[[Any], int] = fn.identity
    decode: Callable[[int], Any] = fn.identity


Encodings = Mapping[str, Encoding]


@attr.s(auto_attribs=True, frozen=True)
class DiscreteCirc:
    circ: BV.AIGBV
    input_encodings: Encodings = attr.ib(
        converter=lambda x: defaultdict(Encoding, x),
    )
    output_encodings: Encodings = attr.ib(
        converter=lambda x: defaultdict(Encoding, x),
    )
    valid_id: str = "##valid"

    def __attrs_post_init__(self):
        if self.valid_id not in self.circ.outputs:
            raise ValueError(f"Missing validation output: {self.valid_id}.")
        elif self.circ.omap[self.valid_id].size != 1:
            raise Valid_Id(f"Validation output must be size 1.")

    @property
    def inputs(self): return self.circ.inputs

    @property
    def outputs(self): return self.circ.outputs - {self.valid_id}

    @property
    def latches(self): return self.circ.latches

    @property
    def latch2init(self): return self.circ.latch2init

    def __call__(self, inputs, latches=None):
        imap = {
            k: self.input_encodings[k].encode(v) for k, v in inputs.items()
        }
        omap, lmap = self.circ(imap, latches=latches)

        valid, *_ = omap[self.valid_id]
        if not valid:
            raise ValueError(f"Invalid inputs: {inputs}")
        del omap[self.valid_id]
        
        omap = {
            k: self.output_encodings[k].decode(v) for k, v in omap.items()
        }
        return omap, lmap

    def __or__(self, other: DiscreteCirc) -> DiscreteCirc:
        pass

    def __lshift__(self, other: DiscreteCirc) -> DiscreteCirc:
        pass

    def __rshift__(self, other: DiscreteCirc) -> DiscreteCirc:
        pass



def from_aigbv(circ: BV.AIGBV, 
               input_encodings: Encodings=None,
               output_encodings: Encodings=None,
               valid_id="##valid") -> DiscreteCirc:
    if input_encodings is None:
        input_encodings = {}
    if output_encodings is None:
        output_encodings = {}
    if valid_id not in circ.outputs:
        circ |= BV.uatom(1, 1).with_output(valid_id).aigbv

    return DiscreteCirc(
        circ=circ,
        input_encodings=input_encodings,
        output_encodings=output_encodings,        
    )


__all__ = ['Encoding', 'DiscreteCirc', 'from_aigbv']
