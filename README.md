# AMP Fingerprint Generation From Simple\_nn

This repository provides functions to generate fingerprints that can be used in AMP using Simple\_nn

See the example directory for a script to do this. I recommend installing this with pip:

```
pip install git+https://github.com/medford-group/amp_simple_nn
```

Then importing the functions and using them similarly to below:

```
from amp_simple_nn.convert import make_amp_descriptors_simple_nn

make_amp_descriptors_simple_nn(images,g2_etas,g2_rs_s,g4_etas,
                               g4_zetas,g4_gammas,cutoff)

```
