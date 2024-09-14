# Ambiqual
Python implementation of Ambiqual full reference objective model as presented in the following paper:

Narbutt M, Skoglund J, Allen A, Chinen M, Barry D, Hines A. AMBIQUAL: Towards a Quality Metric for Headphone Rendered Compressed Ambisonic Spatial Audio. Applied Sciences. 2020; 10(9):3188. https://doi.org/10.3390/app10093188


## Installation
This package can be installed using pip:

`pip install git+https://github.com/QxLabIreland/Ambiqual`

## Usage
The program can be used using the command line tool:

`python -m ambiqual --ref /path/to/dir/reference_signal --deg /path/to/dir/degraded_sginal [--level threshold] [--elc elc] [--ignorefreqbands freq_band]`

## Options
- `ref` - Reference audio file.
- `deg` - Degraded audio file.
- `level` - Intensity binary mask threshold (in dB).
- `elc` - Equal loudness correction mode:
  - `0` - No equal loudness correction.
  - `1` - Equal loudness correction by boosting low and high frequencies.
  - `2` - Equal loudness correction by attenuating low and high frequencies.
- `ignorefreqbands` - Specifies high-frequency bands to ignore (range: 0 to 32):
  - `0` - All 32 frequency bands are considered.
  - `k` - Ignores from the k-th to the 32nd frequency bands in calculations.
 
## Example

The intensity binary map threshold is set to -180dB and equal loudness contours are applied (attenuated low and high-frequency bands):

`python -m ambiqual --ref validation/audiofiles/castanets_fixed_A60_E60_HOA_REF.wav  --deg validation/audiofiles/castanets_fixed_A60_E60_HOA_512k.wav --level -180 --elc 2 --ignorefreqbands 0`

## Validation

To validate Ambiqual, you can use the ambiqual_test.py script located in the validation directory to run Ambiqual on a set of ambisonic audio files. The resulting listening quality and localisation accuracy are then plotted against subjective scores, similar to Figures 12 and 13 in the paper. Note that some audio files were excluded due to copyright, and as a result,  some figures may differ from those in the paper.

<img src="https://github.com/QxLabIreland/Ambiqual/blob/main/validation/fig11.png" alt="Alt text" width="500" height="230">

<img src="https://github.com/QxLabIreland/Ambiqual/blob/main/validation/fig12.png" alt="Alt text" width="1000" height="190">


## Citation

If you use this code, please cite both the repository and the associated paper:

[![DOI](https://zenodo.org/badge/848741719.svg)](https://zenodo.org/doi/10.5281/zenodo.13388476)

Narbutt M, Skoglund J, Allen A, Chinen M, Barry D, Hines A. AMBIQUAL: Towards a Quality Metric for Headphone Rendered Compressed Ambisonic Spatial Audio. Applied Sciences. 2020; 10(9):3188. https://doi.org/10.3390/app10093188

## Licence

This project is licensed under the Apache 2.0 License.
