from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, Optional, Union
from uuid import uuid1

import aiger_bv as BV
import attr
import funcy as fn


def fresh():
    return str(uuid1())


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

    def __or__(self, other: Union[DiscreteCirc, BV.AIGBV]) -> DiscreteCirc:
        other: DiscreteCirc = canon(other)
        both_valid = (self._vexpr & other._vexpr).with_output(self.valid_id)
        circ = (self.circ | other.circ) >> both_valid.aigbv
        return from_aigbv(
            circ=circ,
            input_encodings=fn.merge(other.input_encodings,
                                     self.input_encodings),
            output_encodings=fn.merge(other.output_encodings,
                                      self.output_encodings),
            valid_id=self.valid_id
        )

    def __lshift__(self, other: DiscreteCirc) -> DiscreteCirc:
        pass

    def __rshift__(self, other: DiscreteCirc) -> DiscreteCirc:
        pass


def canon(circ: Union[BV.AIGBV, DiscreteCirc]):
    if not isinstance(circ, DiscreteCirc):
        circ = from_aigbv(circ)
    return circ.rename_valid()


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

    return DiscreteCirc(
        circ=circ,
        input_encodings=input_encodings,
        output_encodings=output_encodings,
    )


__all__ = ['Encoding', 'DiscreteCirc', 'from_aigbv']
