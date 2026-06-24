QUANT_MAX: int = 127   # quantization range [0, QUANT_MAX]
EOS_COORD: int = 128   # face padding token for input sequence
TRI_PAD: int = 129     # padding token of the unified 12-token face block; doubles as
                       # the implicit triangle-type marker.
                       # Whole-face layout — occupies the TRAILING positions 9-11 of a
                       # triangle face (padding moved to the end so positions 0-2 are
                       # always v0):
                       #   [v0_c0, v0_c1, v0_c2, v1_c0, v1_c1, v1_c2,
                       #    v2_c0, v2_c1, v2_c2, TRI_PAD, TRI_PAD, TRI_PAD]
                       # Quad faces carry no padding (all 12 positions are coordinate tokens).
                       # Edge-cond layout — the MLP always predicts two vertices; for a
                       # triangle neighbor the 2nd vertex (slot 2, positions 9-11) is
                       # TRI_PAD.  Same token in input and output: the pad flows from the
                       # input through the MLP, so no dedicated "triangle marker" is needed.
                       # TRI_PAD > QUANT_MAX, so it cannot be confused with a real coordinate.
EOS_RESIDUAL: int = 255  # residual target EOS — slot-1 "STOP, this edge has no neighbor".
                         # Purely a stop signal; trained as a valid class.  In edge-cond
                         # 12-token mode the two prediction slots are:
                         #   slot 1 (positions 6-8)  : new vertex v1, OR EOS_RESIDUAL (STOP)
                         #   slot 2 (positions 9-11) : new vertex v2 (quad), OR TRI_PAD (tri)
                         # EOS_RESIDUAL never appears in slot 2; TRI_PAD never in slot 1.

PAD_TARGET: int = -1   # sentinel for "do not compute loss on this face" (ignored by cross-entropy)
