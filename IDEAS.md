## how to handle distinguishing sequences?

1. apply alibi-style biases to attention scores while keeping everything at position 0

2. restrict certain attention heads to only attend to the current sequence. this seems really elegant. this isn't directly compatible with the flash/flex attention APIs — actually we can maybe use alexzhang13/flashattention2-custom-mask

3. use some kind of sparse mask so that the sequences are mostly isolated except for at select positions.

4. apply a global offset to each parallel sequence (i don't like the asymmetry of this solution).

5. have global attention on select layers

## when is diversity useful for problem solving?

sanity check: generate a bunch of IID samples and evaluate different subsets. plot DPP probability against performance. try different embedding models.

## how to teach diversity?

TBD
