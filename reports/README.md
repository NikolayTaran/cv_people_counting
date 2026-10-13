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
