# DemosaicNet X-Trans golden corpus

These deterministic inference inputs and outputs were generated from the
reviewed Gharbi DemosaicNet X-Trans RTNN artifact. They are small numerical
test fixtures for implementing RawTherapee's native inference engine; they do
not contain the checkpoint or RTNN model weights.

Upstream project: https://github.com/mgharbi/demosaicnet
Upstream revision: 959e9d1630976b421d5af5e35b2e2a01f5630e5c
Paper: Deep Joint Demosaicking and Denoising, SIGGRAPH Asia 2016
License: MIT; see `UPSTREAM-LICENSE.txt`.

The tensors are exact network inputs and outputs. No raw normalization, gamma
wrapper, CFA orientation transform, clipping, sample reinjection, or
postprocessing is represented by this corpus.
