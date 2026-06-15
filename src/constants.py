QUANT_MAX: int = 127   # quantization range [0, QUANT_MAX]
EOS_COORD: int = 128   # face padding token for input sequence
TRI_PAD: int = 129     # triangle-type marker in the unified 12-token face block.
                       # Occupies the TRAILING positions 9-11 of a triangle face
                       # (padding moved to the end so positions 0-2 are always v0):
                       #   [v0_c0, v0_c1, v0_c2, v1_c0, v1_c1, v1_c2,
                       #    v2_c0, v2_c1, v2_c2, TRI_PAD, TRI_PAD, TRI_PAD]
                       # Quad faces carry no padding (all 12 positions are coordinate tokens).
                       # TRI_PAD > QUANT_MAX, so it cannot be confused with a real coordinate.
                       # The model learns face type implicitly from this trailing pad.
EOS_RESIDUAL: int = 255  # residual target EOS (no neighbor to generate) — trained as a valid class

# ── Edge-cond (12-token Option A) slot sentinels — MUST stay distinct ─────────
# In edge-conditioned 12-token mode the model predicts up to two new vertices for
# a query edge, in two slots:
#   slot 1 (positions 6-8)  : new vertex v1, OR EOS_RESIDUAL meaning "STOP — this
#                             edge has no neighbor face".
#   slot 2 (positions 9-11) : new vertex v2 (quad neighbor), OR TRI_NEIGHBOR
#                             meaning "the neighbor is a TRIANGLE — there is no v2".
#
# These two roles MUST use different tokens.  They were historically both encoded
# with EOS_RESIDUAL, which made a single shared softmax class control two
# unrelated decisions ("stop" vs "triangle"): the frequent slot-1 stop signal
# leaked into slot 2 and quads silently collapsed to triangles (quad_recall ≈ 0.7
# while triangles are <5% of neighbors).  Keep STOP (EOS_RESIDUAL, slot 1) and
# TRI_NEIGHBOR (slot 2) as separate sentinels — do NOT merge them again.
#
# TRI_NEIGHBOR=256 reuses the previously-unused output slot of vocab_size=257
# (EOS_RESIDUAL=255 was the highest live class), so no model output dimension
# changes and triangle-only / existing-quad checkpoints stay compatible.
# It is a TARGET-only sentinel: it never appears in input face tokens, and never
# in the 9-token triangle-only path — only in 12-token edge-cond slot-2 targets.
TRI_NEIGHBOR: int = 256

PAD_TARGET: int = -1   # sentinel for "do not compute loss on this face" (ignored by cross-entropy)
