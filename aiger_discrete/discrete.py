from __future__ import annotations

from collections import defaultdict
from functools import reduce
from typing import Any, Callable, Mapping, Optional, Union
from uuid import uuid1

import aiger_bv as BV
import attr
import funcy as fn
from pyrsistent import pmap
from pyrsistent.typing import PMap


def fresh():
    return str(uuid1())


# TODO: implement py-aiger API
@attr.s(auto_attribs=True, frozen=True)
class Encoding:
    encode: Callable[[Any], int] = fn.identity
    decode: Callable[[int], Any] = fn.identity


DefaultEncoding = Encoding()
Encodings = PMap[str, Encoding]


@attr.s(auto_attribs=True, frozen=True)
class DiscreteCirc:
    circ: BV.AIGBV
    input_encodings: Encodings = attr.ib(converter=pmap)
    output_encodings: Encodings = attr.ib(converter=pmap)
    valid_id: str = "##valid"

    def __attrs_post_init__(self):
        if self.valid_id not in self.circ.outputs:
            raise ValueError(f"Missing validation output: {self.valid_id}.")
        elif self.circ.omap[self.valid_id].size != 1:
            raise ValueError("Validation output must be size 1.")

    def rename_valid(self, name: Optional[str] = None) -> DiscreteCirc:
        if name is None:
            name = fresh()
        if name in self.outputs:
            raise ValueError(f"{name} conflicts with current outputs.")
        circ = self.circ['o', {self.valid_id: name}]
        return attr.evolve(self, circ=circ, valid_id=name)

    def assume(self, pred: BV.AIGBV) -> DiscreteCirc:
        assert len(pred.outputs) == 1
        func = from_aigbv(pred.outputs, valid_id=fn.first(pred.outputs))
        return self | func

    @property
    def _vexpr(self):
        return BV.uatom(1, self.valid_id)

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
            k: self.input_encodings.get(k, DefaultEncoding).encode(v)
            for k, v in inputs.items()
        }
        omap, lmap = self.circ(imap, latches=latches)

        valid, *_ = omap[self.valid_id]
        if not valid:
            raise ValueError(f"Invalid inputs: {inputs}")
        del omap[self.valid_id]

        omap = {
            k: self.output_encodings.get(k, DefaultEncoding).decode(v)
            for k, v in omap.items()
        }
        return omap, lmap

    def __or__(self, other: Circ) -> DiscreteCirc:
        other: DiscreteCirc = canon(other)
        circ = (self.circ | other.circ) >> both_valid(self, other)

        # TODO: project to important inputs.
        return from_aigbv(
            circ=circ,
            input_encodings=self.input_encodings + other.input_encodings,
            output_encodings=self.output_encodings + other.output_encodings,
            valid_id=self.valid_id
        )

    def __rshift__(self, other: Circ) -> DiscreteCirc:
        other: DiscreteCirc = canon(other)
        circ = (self.circ >> other.circ) >> both_valid(self, other)
        return from_aigbv(
            circ=circ,
            input_encodings=other.input_encodings + self.input_encodings,
            output_encodings=self.output_encodings + other.output_encodings,
            valid_id=self.valid_id
        )

    def __lshift__(self, other: Circ) -> DiscreteCirc:
        return canon(other) >> self


Circ = Union[DiscreteCirc, BV.AIGBV]


def both_valid(left: DiscreteCirc, right: DiscreteCirc) -> BV.AIGBV:
    return (left._vexpr & right._vexpr).with_output(left.valid_id).aigbv


def canon(circ: Circ) -> DiscreteCirc:
    if not isinstance(circ, DiscreteCirc):
        circ = from_aigbv(circ)
    return circ.rename_valid()


def omit(mapping, keys):
    return reduce(lambda m, k: m.discard(k), keys, mapping)


def project(mapping, keys):
    return omit(mapping, set(mapping.keys()) - keys)


def from_aigbv(circ: BV.AIGBV,
               input_encodings: Encodings = None,
               output_encodings: Encodings = None,
               valid_id="##valid") -> DiscreteCirc:
    if input_encodings is None:
        input_encodings = {}
    if output_encodings is None:
        output_encodings = {}
    if valid_id not in circ.outputs:
        circ |= BV.uatom(1, 1).with_output(valid_id).aigbv

    input_encodings = project(pmap(input_encodings), circ.inputs)
    output_encodings = project(pmap(output_encodings), circ.outputs - {valid_id})

    return DiscreteCirc(
        circ=circ,
        input_encodings=input_encodings,
        output_encodings=output_encodings,
    )


__all__ = ['Encoding', 'DiscreteCirc', 'from_aigbv']
