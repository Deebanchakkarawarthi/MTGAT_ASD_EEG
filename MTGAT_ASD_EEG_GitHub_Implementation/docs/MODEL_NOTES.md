# Model implementation notes

- `plv_matrix` computes frequency-band PLV from channel x sample epochs.
- `DifferentiableDynamicPLV` learns temporal weighting over cosine/sine phase-difference features; analytic phase extraction is external.
- `MTGAT` accepts `[batch, time, bands, nodes, features]` and adjacency `[batch, time, bands, nodes, nodes]`.
- The feature extractor emits one graph per epoch; temporal sequences must be assembled explicitly for the GRU model.
- The current extractor uses channel mean absolute amplitude repeated across bands as a baseline feature, not true band-specific power. Replace it before final experiments.
- The example trainer optimizes ASD classification only. Multi-task heads require verified labels and explicit loss terms.
- Dense attention currently uses positive adjacency as a mask rather than a PLV additive bias.
- No clinical validation or performance results are claimed by this repository.
