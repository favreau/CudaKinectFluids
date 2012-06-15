#ifndef DEFINES_H
#define DEFINES_H

#ifdef __DEVICE_EMULATION__
#define DIM    64        // Square size of solver domain
#else
#define DIMX  512       // Square size of solver domain
#define DIMY  512       // Square size of solver domain
#endif
#define DS    (DIMX*DIMY)  // Total domain size
#define CPADW (DIMX/2+1)  // Padded width for real->complex in-place FFT
#define RPADW (2*(DIMX/2+1))  // Padded width for real->complex in-place FFT
#define PDS   (DIMX*CPADW) // Padded total domain size

#define DT     0.09f     // Delta T for interative solver
#define VIS    0.0025f   // Viscosity constant
#define FORCE (5.8f*DIMX) // Force scale factor 
#define FR     4         // Force update radius

#define TILEX 64 // Tile width
#define TILEY 64 // Tile height
#define TIDSX 64 // Tids in X
#define TIDSY 4  // Tids in Y

#endif
