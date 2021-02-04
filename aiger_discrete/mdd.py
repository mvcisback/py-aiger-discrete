"""Converters to and from multi-valued decision diagrams."""

from functools import reduce
from typing import Optional

import aiger_bv as BV
import funcy as fn
import mdd
from aiger_bv.bundle import Bundle

from aiger_discrete import Encoding, FiniteFunc


def to_var(bdl: Bundle, encoding: Optional[Encoding]) -> mdd.Variable:
    if encoding is None:
        encoding = Encoding()

    return mdd.Variable(
        encode=encoding.encode,
        decode=encoding.decode,
        valid=BV.uatom(bdl.size, bdl.name)[0] | 1,  # const 1.
    )


def onehot_output(expr):
    """Creates circuit that depends only on 1-hot active bit."""
    bits = BV.uatom(expr.size, expr.output)

    def ite(test, idx):
        return BV.ite(expr[idx], bits[idx], test)

    # Create chained if then else testing 1-hot bit.
    return reduce(ite, range(1, expr.size), bits[0])


def to_mdd(func: FiniteFunc, manager=None, order=None) -> mdd.DecisionDiagram:
    reordering_allowed = order is None
    if order is None:
        order = func.circ.inputs
    order = list(order)

    # TODO: currently assuming 1-hot.
    assert len(func.outputs) == 1
    output = fn.first(func.outputs)
    expr = BV.UnsignedBVExpr(func.circ.cone(output))
    output = to_var(func.circ.omap[output], func.output_encodings.get(output))

    imap = func.circ.imap
    input_encodings = func.input_encodings

    inputs = {k: to_var(imap[k], input_encodings.get(k)) for k in order}

    interface = mdd.Interface(
        inputs=inputs,
        output=output,
        valid=BV.UnsignedBVExpr(func.circ.cone(func.valid_id)),
    )

    dd = interface.lift(onehot_output(expr), order=order + [output.name])
    if not reordering_allowed:
        dd.bdd.bdd.configure(reordering=False)
    return dd
