QUANT_MAX: int = 127   # quantization range [0, QUANT_MAX]
EOS_COORD: int = 128   # face padding token for input sequence
TRI_PAD: int = 129     # triangle-type marker in the unified 12-token face block.
                       # Occupies positions 0-2 of a triangle face:
                       #   [TRI_PAD, TRI_PAD, TRI_PAD, v0_c0, v0_c1, v0_c2,
                       #                               v1_c0, v1_c1, v1_c2,
                       #                               v2_c0, v2_c1, v2_c2]
                       # Quad faces carry no padding (all 12 positions are coordinate tokens).
                       # TRI_PAD > QUANT_MAX, so it cannot be confused with a real coordinate.
                       # The model learns face type implicitly from this prefix.
EOS_RESIDUAL: int = 255  # residual target EOS (no neighbor to generate) — trained as a valid class
PAD_TARGET: int = -1   # sentinel for "do not compute loss on this face" (ignored by cross-entropy)
