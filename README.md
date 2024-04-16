# Automated Extraction of Prosodic Structure from Unannotated Sign Language Video

This repository contains the code for the titular article, presented at
LREC-COLING 2024.

Apart from the methodology itself, a command line tool is included that makes
reproducing our experiments (hopefully) easy, and allows using our code to
extract some information from unannotated video.

This code is free and open source software, licensed under the EUPL.

## Features

- Command line interface
- Install with pip: `pip install sign-prosody-extraction`
- Plot the prosodic structure (velocity and direction profile of articulators
    along the temporal axis) of a sign language video.
- Extract thumbnails at "target points": points of low velocity, i.e. high
    static visual significance.
- Extract a short clip that includes only the sign articulation, ignoring
    preparation and relaxation. This can help dictionary makers prepare high
    significance animated thumbnails of their videos.
- Use either cotracker or mediapipe, as outlined in Börstell, 2023, for the
    articulator tracking.

## Usage

    pip install sign-prosody-extraction
    # To compute everythig for VIDEO.mp4, and save in output/
    sign-prosody-extraction VIDEO.mp4 --output-dir output/ --everything
    # To see the different options
    sign-prosody-extraction --help

You can modify some further runtime options with environment variables. Set
`CACHE_DIR` and optionally `CACHE_LIMIT` to cache some long computations.
To use the mediapipe algorithm, the `pose_landmarker.task` model must be
downloaded, and placed in the `data` folder or in a path pointed to by the env
var `POSE_LANDMARKER`. The cotracker model will be loaded from torchhub.

## Article Abstract

As in oral phonology, prosody is an important carrier of linguistic information
in sign languages. One of the most prominent ways this reveals itself is in the
time structure of signs: their rhythm and intensity of articulation. To be able
to empirically see these effects, the velocity of the hands can be computed
throughout the execution of a sign. In this article, we propose a method for
extracting this information from unlabeled videos of sign language, exploiting
CoTracker, a recent advancement in computer vision which can track every point
in a video without the need of any calibration or fine-tuning. The dominant hand
is identified via clustering of the computed point velocities, and its dynamic
profile plotted to make apparent the prosodic structure of signing. We apply our
method to different datasets and sign languages, and perform a preliminary
visual exploration of results. This exploration supports the usefulness of our
methodology for linguistic analysis, though issues to be tackled remain, such as
bi-manual signs and a formal and numerical evaluation of accuracy. Nonetheless,
the absence of any preprocessing requirements may make it useful for other
researchers and datasets.

- Read more at the LREC-COLING 2024 proceedings (coming soon)

## How to cite

If you use our code, please do cite us! Also don't hesitate to get in touch.

Bibtex code: coming soon!

## Authors

<table>
<tr><td align="center">

[![](https://github.com/agarsev.png?size=100)](https://github.com/agarsev) <br>
Antonio F. G. Sevilla <br>
<antonio@garciasevilla.com>

</td><td align="center">

[![](https://github.com/jmlahoz.png?size=100)](https://github.com/jmlahoz) <br>
José María Lahoz Bengoechea <br>
<jmlahoz@ucm.es>

</td><td align="center">

Alberto Díaz Esteban <br>
<albertodiaz@fdi.ucm.es>

</td></tr>
</table>

## Related software

- [cotracker](https://github.com/facebookresearch/co-tracker)
- [mediapipe](https://github.com/google/mediapipe)
- [SciPy](https://github.com/scipy/scipy)
- [Scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [Rye](https://github.com/astral-sh/rye)
- [Click](https://github.com/pallets/click/)
