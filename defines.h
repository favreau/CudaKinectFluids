/* Copyright (c) 2011-2012, Cyrille Favreau
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
 *
 * This file is part of OpenCLKinectFluids
 * <https://github.com/favreau/OpenCLKinectFluids>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef DEFINES_H
#define DEFINES_H

#ifdef __DEVICE_EMULATION__
#define DIM 64 // Square size of solver domain
#else
#define DIMX 512 // Square size of solver domain
#define DIMY 512 // Square size of solver domain
#endif
#define DS (DIMX * DIMY)     // Total domain size
#define CPADW (DIMX / 2 + 1) // Padded width for real->complex in-place FFT
#define RPADW                                                                  \
  (2 * (DIMX / 2 + 1))     // Padded width for real->complex in-place FFT
#define PDS (DIMX * CPADW) // Padded total domain size

#define DT 0.09f            // Delta T for interative solver
#define VIS 0.0025f         // Viscosity constant
#define FORCE (5.8f * DIMX) // Force scale factor
#define FR 4                // Force update radius

#define TILEX 64 // Tile width
#define TILEY 64 // Tile height
#define TIDSX 64 // Tids in X
#define TIDSY 4  // Tids in Y

#define gKinectVideoWidth 640
#define gKinectVideoHeight 480
#define gKinectVideo 4

#define gKinectDepthWidth 320
#define gKinectDepthHeight 240
#define gKinectDepth 2

#endif
