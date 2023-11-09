### MicroTorch Library
Library of PyTorch implementations of microstructural and quantitative MRI models.

Cardiff University Brain Research Imaging Centre

## Dependencies (incomplete)
PyTorch
Numpy

### Grad file

The grad file encodes the MR acquisition parameters for each imaging volume. The format is essentially an MRtrix-style grad file with extra columns. Each row corresponds to one imaging volume, and contains space-separated values [x y z b DELTA, delta, G, TE, TI, TR], where

x, y, z are the diffusion gradient directions
b is the b-value
DELTA is the time between the two gradient pulses
delta is the pulse gradient duration
TE is the echo time
TI is the inversion time
TR is the repitition time

###File name conventions


###Main functions


###Example

###Citations

If you use the code for any of the models then please cite the appropriate paper:

Compartment models: Panagiotaki, Eleftheria, et al. "Compartment models of the diffusion MR signal in brain white matter: a taxonomy and comparison." Neuroimage 59.3 (2012): 2241-2254.

Ball and Stick: Behrens, Timothy EJ, et al. "Characterization and propagation of uncertainty in diffusion‐weighted MR imaging." Magnetic Resonance in Medicine 50.5 (2003): 1077-1088.

VERDICT (BallSphereStick): Panagiotaki, Eletheria, et al. "Noninvasive quantification of solid tumor microstructure using VERDICT MRI." Cancer research 74.7 (2014): 1902-1912.

VERDICT (BallSphereAstrosticks):Panagiotaki, Eleftheria, et al. "Microstructural characterization of normal and malignant human prostate tissue with vascular, extracellular, and restricted diffusion for cytometry in tumours magnetic resonance imaging." Investigative radiology 50.4 (2015): 218-227.

SANDI (BallAstrosticksSphere): Palombo, Marco, et al. "SANDI: a compartment-based model for non-invasive apparent soma and neurite imaging by diffusion MRI." Neuroimage 215 (2020): 116835.

