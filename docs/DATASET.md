# Dataset source and use notes

## Selected dataset

Prathum Arikeri, [Indian Sign Language (ISL), character-level](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl), Kaggle dataset version 1 (published 2021-06-04).

Kaggle lists the dataset under **CC BY-SA 4.0**. Credit the creator and source,
and follow the ShareAlike terms when distributing adaptations. The public
download is about 294 MB and contains still images for letters A-Z and digits
1-9; digit 0 is omitted. This workspace download contained 42,745 images in 35
label folders.

## Provenance and label limitations

The dataset creator says the collection combines their own images with images
from other Kaggle and GitHub datasets. Review the Kaggle acknowledgements and
upstream sources before redistribution. Public availability alone does not
establish reuse rights for every included image. Ask an ISL-knowledgeable
reviewer to check samples from every class before claiming that the labels are
validated ISL.

No signer or session identifiers are supplied in the downloaded class-folder
structure. The project therefore reports an image-level baseline, which may
overstate performance on new signers or recording conditions.

## Expected folder layout

The downloader extracts the dataset under `data/isl_alphabet/`; the labels may
be nested one level below `Indian/`:

```text
data/isl_alphabet/Indian/A/*.jpg
data/isl_alphabet/Indian/B/*.jpg
```

The loader also supports `train/<label>/image.jpg` and
`test/<label>/image.jpg`. Use ISL classes only; do not substitute ASL data.
