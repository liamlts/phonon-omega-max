# Physics findings

The Magpie baseline sees only the composition; the CGCNN sees both
composition and the crystal structure. Materials where CGCNN beats
GBDT by more than 50 cm⁻¹ test the question: when does structure
contain decisive information that composition alone cannot supply?

The expected pattern is that polymorphs (same composition, different
structure) are the cleanest case — GBDT must predict the average of
their ω_max values, while CGCNN can distinguish them.

## (filled in after run)

For each pattern observed, a 2–4 sentence write-up grounded in
concrete material examples (mp_id and reduced formula).
