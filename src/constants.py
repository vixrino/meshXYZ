QUANT_MAX: int = 127   # quantization range [0, QUANT_MAX]
EOS_COORD: int = 128   # face padding token for input sequence
EOS_RESIDUAL: int = 255  # residual target EOS (no neighbor to generate) — trained as a valid class
PAD_TARGET: int = -1   # sentinel for "do not compute loss on this face" (ignored by cross-entropy)
