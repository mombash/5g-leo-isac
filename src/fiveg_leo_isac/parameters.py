"""Fixed units and numerical support of the selected implementation."""
import numpy as np
FS=30.72e6
FC=2.190e9
C=299792458.
BLOCK=1536
DELAYS=1792
MIN_DELAY=80
FMAX=1600.
COUNT=int(.02*FS)
GUARD=int(.01*FS)
PER=COUNT//BLOCK
TIMES=np.array([.4,.5,.6,.7,.8,.9,1.])
FREQUENCIES=np.arange(-FMAX,FMAX+.1,25.,dtype=np.float32)
RATES=np.arange(-52.,52.1,4.)
