# Interpretation of recorded results

The JSON files whose names end in `_test.json` were generated with the
pipeline's `--split test` option. The four sequences in that code split were
excluded from detector training, but their counting results were compared to
choose the submitted 3-epoch checkpoint.

Consequently, `0.6548` mean counting accuracy for the submitted model is a
**post-selection comparison result**. It should not be presented as an
unbiased final test estimate. An independent estimate requires freezing the
current pipeline and evaluating it once on new, untouched cameras.

See [the full methodology](../docs/FIRST_WORKING_MODEL.md) for the split,
training setup, per-sequence counts, and limitations.

The four `second_approach_*.json` files record matched reruns of the single
detector and the two-detector ensemble on both development and camera-separated
comparison sequences. See [the second approach report](../docs/SECOND_APPROACH.md)
for results and interpretation. The development videos contain detector training
frames; the comparison videos were used during earlier model selection. Neither
set is an untouched final test.
