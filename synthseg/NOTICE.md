# Notice of modification: SynthSeg 2.0 networks

`synthseg_robust_2.0.onnx` and `synthseg_parc_2.0.onnx` are modified versions of
`synthseg_robust_2.0.h5` and `synthseg_parc_2.0.h5`, which are part of FreeSurfer
(© The General Hospital Corporation, Boston, MA, USA) and distributed under the FreeSurfer Software
License Agreement, reproduced in `../LICENSES/FreeSurfer.txt`.

What was changed:

- **Format**: Keras HDF5 → ONNX (opset 17) with `tf2onnx`. Architecture and weights are unchanged.
- **Graph split**: FreeSurfer's `mri_synthseg --robust --parc` runs both networks as one graph.
  Here the robust segmentation and the cortical parcellation are exported as two files; the
  parcellation file takes the image plus a background/cortex one-hot as its 3-channel input, which
  `mri_synthseg` computes inside its graph and `leaddbs` computes outside.
- **Metadata**: each file carries its label list and the SHA-256 of the HDF5 file it came from.

Verified equivalent on one reference subject: with FreeSurfer's own pre- and post-processing,
swapping in the ONNX network changes 64 of about 1.7 million brain voxels (0.004%; every one on a
label boundary; minimum per-label Dice 0.9994), from onnxruntime and TensorFlow rounding
differently at near-ties.

SynthSeg is by B. Billot, D. N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl,
A. V. Dalca and J. E. Iglesias. This repository is not affiliated with or endorsed by the FreeSurfer
project or MGH.
