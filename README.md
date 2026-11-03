# leaddbs-models

Pre-trained networks that Lead-DBS 4.0 (`leaddbs` on PyPI) downloads on first use, converted to
ONNX so they run under `onnxruntime` without TensorFlow or PyTorch. The files are published as
release assets, never committed; `leaddbs` pins each one by name and SHA-256.

Releases are per model version (the first: `synthseg-2.0`); `leaddbs` downloads
`https://github.com/JPPayonk/leaddbs-models/releases/download/<release>/<file>`.

## Contents

| File | Network | Source weights | Licence |
| --- | --- | --- | --- |
| `synthseg_robust_2.0.onnx` | SynthSeg-robust 2.0: T1 (any contrast) → 33 aseg posteriors | FreeSurfer `synthseg_robust_2.0.h5` | FreeSurfer, modified |
| `synthseg_parc_2.0.onnx` | SynthSeg 2.0 cortical parcellation → 69 aparc posteriors | FreeSurfer `synthseg_parc_2.0.h5` | FreeSurfer, modified |

Digests of the files and of the weights they were converted from: `synthseg/manifest.json`, a copy
of the `dist/manifest.json` the conversion writes.

## Licences

The **model files** are derived from weights distributed with FreeSurfer and remain under the
[FreeSurfer Software License Agreement](LICENSES/FreeSurfer.txt). They are modified versions in the
sense of that agreement: the networks and weights are unchanged, but converted from Keras HDF5 to
ONNX, and each SynthSeg network graph is exported separately. See [`synthseg/NOTICE.md`](synthseg/NOTICE.md).

The **conversion scripts** in this repository are [GPL-3.0-or-later](LICENSES/GPL-3.0-or-later.txt), like `leaddbs`.

Each `.onnx` file also carries that licence, with the preface its paragraph B.1(b) requires, in its
ONNX metadata (`license_text`), so the terms travel with every copy of the file.

These files are not part of the `leaddbs` wheel and are not covered by its licence.

**Research use only.** As the FreeSurfer licence states, the networks are designed for research
purposes only and have not been reviewed or approved by the FDA or any other agency; clinical
applications are neither recommended nor advised.

## Reproducing the files

```bash
python synthseg/convert.py --mri-synthseg <leaddbs-3.2.1>/ext_libs/SynthSeg/mri_synthseg --out dist/
```

in Lead-DBS 3.2.1's `synthseg` conda environment (TensorFlow 2.13) with `tf2onnx==1.16.1` and
`onnx==1.16.2` added, e.g. through a `--system-site-packages` venv on top of it. The script builds
each graph with FreeSurfer's own `mri_synthseg`, so the export is the network that script runs.

## Citing

SynthSeg: B. Billot et al., *Medical Image Analysis* 2023. Robust version: B. Billot et al.,
*PNAS* 2023.
