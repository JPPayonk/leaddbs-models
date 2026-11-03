"""Convert the SynthSeg 2.0 robust and cortical-parcellation networks from Keras .h5 to ONNX.

The graphs are built by FreeSurfer's own ``mri_synthseg`` (as vendored in Lead-DBS 3.2.1), so each
ONNX file is the network that script runs, not a re-implementation of it. Two files, because the
parcellation is optional and costs about 75% more time and 6.6 GB more memory:

* ``synthseg_robust_2.0.onnx``: T1 patch -> 33 segmentation posteriors
* ``synthseg_parc_2.0.onnx``: [T1 patch, background/cortex one-hot] -> 69 parcellation posteriors

Each file carries its label lists and provenance in its ONNX metadata.

Environment: Lead-DBS 3.2.1's ``synthseg`` conda env (TensorFlow 2.13, Keras 2.13) plus
``tf2onnx==1.16.1 onnx==1.16.2``, e.g. a ``--system-site-packages`` venv on top of it.

    python convert.py --mri-synthseg ~/git/leaddbs/ext_libs/SynthSeg/mri_synthseg --out dist/
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path

import numpy as np

OPSET = 17


def load_script(path: Path):
    loader = importlib.machinery.SourceFileLoader("mri_synthseg", str(path))
    spec = importlib.util.spec_from_loader("mri_synthseg", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export(model, input_channels: int, out: Path, metadata: dict[str, str]) -> None:
    import onnx
    import tensorflow as tf
    import tf2onnx

    spec = (tf.TensorSpec((1, None, None, None, input_channels), tf.float32, name="image"),)
    proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=OPSET)
    for key, value in metadata.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(proto)
    onnx.save(proto, str(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mri-synthseg", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ms = load_script(args.mri_synthseg)
    from tensorflow import keras

    home = Path(ms.models_home)
    robust_h5 = home / "synthseg_robust_2.0.h5"
    parc_h5 = home / "synthseg_parc_2.0.h5"
    # robust mode uses np.unique-sorted labels and no left/right flip (as predict() does)
    labels_seg = np.unique(ms.get_list_labels(str(home / "synthseg_segmentation_labels_2.0.npy"))[0])
    labels_den = np.unique(ms.get_list_labels(str(home / "synthseg_denoiser_labels_2.0.npy"))[0])
    labels_parc = np.unique(ms.get_list_labels(str(home / "synthseg_parcellation_labels.npy"))[0])
    names_seg = np.load(home / "synthseg_segmentation_names_2.0.npy")
    _, first = np.unique(ms.get_list_labels(str(home / "synthseg_segmentation_labels_2.0.npy"))[0],
                         return_index=True)
    names_seg = names_seg[first]

    robust = ms.build_model(
        model_file_segmentation=str(robust_h5), model_file_parcellation=None, model_file_qc=None,
        labels_segmentation=labels_seg, labels_denoiser=labels_den, labels_parcellation=None,
        labels_qc=None, flip_indices=None, robust=True, do_parcellation=False, do_qc=False,
    )
    parc = ms.unet(nb_features=24, input_shape=[None, None, None, 3], nb_levels=5, conv_size=3,
                   nb_labels=len(labels_parc), feat_mult=2, activation="elu", nb_conv_per_level=2,
                   batch_norm=-1, name="unet_parc")
    parc.load_weights(str(parc_h5), by_name=True)
    parc = keras.Model(parc.inputs, ms.GaussianBlur(sigma=0.5)(parc.output))

    common = {
        "source": "SynthSeg 2.0 (Billot et al.) weights from FreeSurfer, converted to ONNX",
        "license": "FreeSurfer Software License Agreement; modified (format conversion)",
        # The licence's B.1(b): its terms, with its preface, appear in every copy -- in the file,
        # since a release asset travels without the repository around it.
        "license_text": (Path(__file__).parents[1] / "LICENSES" / "FreeSurfer.txt").read_text(),
        "converter": f"tf2onnx opset {OPSET}",
    }
    jobs = [
        ("synthseg_robust_2.0.onnx", robust, 1, robust_h5,
         {"labels": json.dumps(labels_seg.tolist()), "names": json.dumps(names_seg.tolist())}),
        ("synthseg_parc_2.0.onnx", parc, 3, parc_h5,
         {"labels": json.dumps(labels_parc.tolist())}),
    ]
    manifest = {}
    for name, model, channels, source, meta in jobs:
        out = args.out / name
        export(model, channels, out, {**common, **meta, "source_sha256": sha256(source)})
        manifest[name] = {"sha256": sha256(out), "bytes": out.stat().st_size,
                          "source": source.name, "source_sha256": sha256(source)}
        print(f"{name}: {out.stat().st_size / 2**20:.0f} MiB")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
