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

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "cutil_inline.h"

// CUDA FFT Libraries
#include <cufft.h>

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// FluidsGL CUDA kernel definitions
#include "fluidsGL_kernels.cuh"

// Texture reference for reading velocity field
texture<float2, 2> texref;
static cudaArray *array = NULL;

// Particle data
extern GLuint vbo;                 // OpenGL vertex buffer object
extern struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange

// Texture pitch
extern size_t tPitch;
extern cufftHandle planr2c;
extern cufftHandle planc2r;
extern cData *vxfield = NULL;
extern cData *vyfield = NULL;

__constant__ __device__ BYTE* d_kinectVideo;
__constant__ __device__ BYTE* d_kinectDepth;

void setupTexture(int x, int y) {
   // Wrap mode appears to be the new default
   texref.filterMode = cudaFilterModeLinear;
   cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

   cudaMallocArray(&array, &desc, y, x);
   cutilCheckMsg("cudaMalloc failed");
}

void bindTexture(void) {
   cudaBindTextureToArray(texref, array);
   cutilCheckMsg("cudaBindTexture failed");
}

void unbindTexture(void) {
   cudaUnbindTexture(texref);
}

void updateTexture(cData *data, size_t wib, size_t h, size_t pitch) {
   cudaMemcpy2DToArray(array, 0, 0, data, pitch, wib, h, cudaMemcpyDeviceToDevice);
   cutilCheckMsg("cudaMemcpy failed"); 
}

void deleteTexture(void) {
   cudaFreeArray(array);
}

#if 0
// Note that these kernels are designed to work with arbitrary 
// domain sizes, not just domains that are multiples of the tile
// size. Therefore, we have extra code that checks to make sure
// a given thread location falls within the domain boundaries in
// both X and Y. Also, the domain is covered by looping over
// multiple elements in the Y direction, while there is a one-to-one
// mapping between threads in X and the tile size in X.
// Nolan Goodnight 9/22/06

// This method adds constant force vectors to the velocity field 
// stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
__global__ void 
   addForces_k(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch) {

      int tx = threadIdx.x;
      int ty = threadIdx.y;
      cData *fj = (cData*)((char*)v + (ty + spy) * pitch) + tx + spx;

      cData vterm = *fj;
      tx -= r; ty -= r;
      float s = 1.f / (1.f + tx*tx*tx*tx + ty*ty*ty*ty);
      vterm.x += s * fx;
      vterm.y += s * fy;
      *fj = vterm;
}

// This method performs the velocity advection step, where we
// trace velocity vectors back in time to update each grid cell.
// That is, v(x,t+1) = v(p(x,-dt),t). Here we perform bilinear
// interpolation in the velocity space.
__global__ void 
   advectVelocity_k(cData *v, float *vx, float *vy,
   int dx, int pdx, int dy, float dt, int lb) {

      int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
      int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
      int p;

      cData vterm, ploc;
      float vxterm, vyterm;
      // gtidx is the domain location in x for this thread
      if (gtidx < dx) {
         for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
               int fj = fi * pdx + gtidx;
               vterm = tex2D(texref, (float)gtidx, (float)fi);
               ploc.x = (gtidx + 0.5f) - (dt * vterm.x * dx);
               ploc.y = (fi + 0.5f) - (dt * vterm.y * dy);
               vterm = tex2D(texref, ploc.x, ploc.y);
               vxterm = vterm.x; vyterm = vterm.y; 
               vx[fj] = vxterm;
               vy[fj] = vyterm; 
            }
         }
      }
}

// This method performs velocity diffusion and forces mass conservation 
// in the frequency domain. The inputs 'vx' and 'vy' are complex-valued 
// arrays holding the Fourier coefficients of the velocity field in
// X and Y. Diffusion in this space takes a simple form described as:
// v(k,t) = v(k,t) / (1 + visc * dt * k^2), where visc is the viscosity,
// and k is the wavenumber. The projection step forces the Fourier
// velocity vectors to be orthogonal to the vectors for each
// wavenumber: v(k,t) = v(k,t) - ((k dot v(k,t) * k) / k^2.
__global__ void 
   diffuseProject_k(cData *vx, cData *vy, int dx, int dy, float dt, 
   float visc, int lb) {

      int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
      int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
      int p;

      cData xterm, yterm;
      // gtidx is the domain location in x for this thread
      if (gtidx < dx) {
         for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
               int fj = fi * dx + gtidx;
               xterm = vx[fj];
               yterm = vy[fj];

               // Compute the index of the wavenumber based on the
               // data order produced by a standard NN FFT.
               int iix = gtidx;
               int iiy = (fi>dy/2)?(fi-(dy)):fi;

               // Velocity diffusion
               float kk = (float)(iix * iix + iiy * iiy); // k^2 
               float diff = 1.f / (1.f + visc * dt * kk);
               xterm.x *= diff; xterm.y *= diff;
               yterm.x *= diff; yterm.y *= diff;

               // Velocity projection
               if (kk > 0.f) {
                  float rkk = 1.f / kk;
                  // Real portion of velocity projection
                  float rkp = (iix * xterm.x + iiy * yterm.x);
                  // Imaginary portion of velocity projection
                  float ikp = (iix * xterm.y + iiy * yterm.y);
                  xterm.x -= rkk * rkp * iix;
                  xterm.y -= rkk * ikp * iix;
                  yterm.x -= rkk * rkp * iiy;
                  yterm.y -= rkk * ikp * iiy;
               }

               vx[fj] = xterm;
               vy[fj] = yterm;
            }
         }
      }
}

// This method updates the velocity field 'v' using the two complex 
// arrays from the previous step: 'vx' and 'vy'. Here we scale the 
// real components by 1/(dx*dy) to account for an unnormalized FFT. 
__global__ void 
   updateVelocity_k(cData *v, float *vx, float *vy, 
   int dx, int pdx, int dy, int lb, size_t pitch) {

      int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
      int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
      int p;

      float vxterm, vyterm;
      cData nvterm;
      // gtidx is the domain location in x for this thread
      if (gtidx < dx) {
         for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
               int fjr = fi * pdx + gtidx; 
               vxterm = vx[fjr];
               vyterm = vy[fjr];

               // Normalize the result of the inverse FFT
               float scale = 1.f / (dx * dy);
               nvterm.x = vxterm * scale;
               nvterm.y = vyterm * scale;

               cData *fj = (cData*)((char*)v + fi * pitch) + gtidx;
               *fj = nvterm;
            }
         } // If this thread is inside the domain in Y
      } // If this thread is inside the domain in X
}

// This method updates the particles by moving particle positions
// according to the velocity field and time step. That is, for each
// particle: p(t+1) = p(t) + dt * v(p(t)).  
__global__ void 
   advectParticles_k(cData *part, cData *v, int dx, int dy, 
   float dt, int lb, size_t pitch) {

      int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
      int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
      int p;

      // gtidx is the domain location in x for this thread
      cData pterm, vterm;
      if (gtidx < dx) {
         for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
               int fj = fi * dx + gtidx;
               pterm = part[fj];

               int xvi = ((int)(pterm.x * dx));
               int yvi = ((int)(pterm.y * dy));
               vterm = *((cData*)((char*)v + yvi * pitch) + xvi);   

               pterm.x += dt * vterm.x;
               pterm.x = pterm.x - (int)pterm.x;            
               pterm.x += 1.f; 
               pterm.x = pterm.x - (int)pterm.x;              
               pterm.y += dt * vterm.y;
               pterm.y = pterm.y - (int)pterm.y;            
               pterm.y += 1.f; 
               pterm.y = pterm.y - (int)pterm.y;                  

               part[fj] = pterm;
            }
         } // If this thread is inside the domain in Y
      } // If this thread is inside the domain in X
}
#endif // 0

// This method updates the particles by moving particle positions
// according to the velocity field and time step. That is, for each
// particle: p(t+1) = p(t) + dt * v(p(t)).  
__global__ void 
   feelTheAttraction_k(cData *particules, float3 position, float timer, float param, BYTE* depth) 
{
   int index = blockDim.x*blockIdx.x + threadIdx.x;
   float3 originalPosition;
   originalPosition.y = (index / DIMX) ;
   originalPosition.x = (index - (originalPosition.y*DIMX));
   originalPosition.z = 0.f;
   
   /*
   int depthIndex = originalPosition.y/(512/gKinectDepthHeight)*gKinectDepthWidth+0.3f*originalPosition.x/(512/gKinectDepthWidth);
   int di = depthIndex*gKinectDepth;
   */
   float d = ((512*512)/(gKinectDepthWidth*gKinectDepthHeight*gKinectDepth));
   int  di = gKinectDepth*index*d;

   float depthPosition = 0.f;
   unsigned char p = 0;
   if( di < gKinectDepthWidth*gKinectDepthHeight*gKinectDepth) 
   {
      unsigned char a = depth[di];
      unsigned char b = depth[di+1];
      p = a & 7;
      unsigned char A = (a<<3);
      depthPosition = (A*256.f+b)/1024.f;
   }
   
   //if( position.x != 0.f && position.y != 0.f )
   if( p != 0 )
   {
      float3 length;
      length.x = (position.x - particules[index].x+0.5f);
      length.y = (position.y - particules[index].y+0.5f);
      length.z = (position.z - particules[index].z+0.5f);

      float l = (depthPosition+sqrt(length.x*length.x + length.y*length.y + length.z*length.z))/param;

      //if( l > 0.7f*param )
      {
         particules[index].x += l/length.x + 0.005f*cos(timer*32.f+l);
         particules[index].y += l/length.y + 0.005f*sin(timer*22.f+l);
      }
      /*
      else
      {
         if( l > 0.69f*param )
         {
            particules[index].x += 0.5f*cos(timer+l);
            particules[index].y += 0.5f*sin(timer+l);
         }
         else
         {
            particules[index].x -= length.x/l;
            particules[index].y -= length.y/l;
         }
      }
      */
   }
   else 
   {
      particules[index].x += (originalPosition.x/DIMX - particules[index].x)/10.f;
      particules[index].y += (originalPosition.y/DIMY - particules[index].y)/10.f;
      //particules[index].z += (originalPosition.z/DIMX - particules[index].z)/10.f;
   }
}

// These are the external function calls necessary for launching fluid simuation
extern "C" void feelTheAttraction(
   cData *particules, int dx, int dy, float3 position, float timer, float param ) 
{ 
   cData *p;
   cutilSafeCall(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
   size_t num_bytes; 
   cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&p, &num_bytes, cuda_vbo_resource));
   cutilCheckMsg("cudaGraphicsResourceGetMappedPointer failed");

   dim3 gridSize(512,1,1);
   dim3 blockSize(dx*dy/gridSize.x,1,1);
   
   feelTheAttraction_k<<<gridSize, blockSize>>>(p,position,timer, param, d_kinectDepth);
   //cutilCheckMsg("feelTheAttraction_k failed.");

   cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

   cutilCheckMsg("cudaGraphicsUnmapResources failed");
}

extern "C" void h2d_kinect( BYTE* kinectVideo, BYTE* kinectDepth )
{
   //cutilSafeCall(cudaMemcpy( d_kinectVideo, kinectVideo, gKinectVideoWidth*gKinectVideoHeight*gKinectVideo*sizeof(BYTE), cudaMemcpyHostToDevice ));
   cutilSafeCall(cudaMemcpy( d_kinectDepth, kinectDepth, gKinectDepthWidth*gKinectDepthHeight*gKinectDepth*sizeof(BYTE), cudaMemcpyHostToDevice ));
}

extern "C" void initialize_scene()
{
   //cutilSafeCall(cudaMalloc( (void**)&d_kinectVideo, gKinectVideoWidth*gKinectVideoHeight*gKinectVideo*sizeof(BYTE)));
   cutilSafeCall(cudaMalloc( (void**)&d_kinectDepth, gKinectDepthWidth*gKinectDepthHeight*gKinectDepth*sizeof(BYTE)));
}

extern "C" void finalize_scene()
{
   //cutilSafeCall(cudaFree( d_kinectVideo ));
   cutilSafeCall(cudaFree( d_kinectDepth ));
}

#if 0
// These are the external function calls necessary for launching fluid simuation
extern "C"
   void addForces(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r) { 

      dim3 tids(2*r+1, 2*r+1);

      addForces_k<<<1, tids>>>(v, dx, dy, spx, spy, fx, fy, r, tPitch);
      cutilCheckMsg("addForces_k failed.");
}

extern "C"
   void advectVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy, float dt) 
{ 
   dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));

   dim3 tids(TIDSX, TIDSY);

   updateTexture(v, DIMX*sizeof(cData), DIMX, tPitch);
   advectVelocity_k<<<grid, tids>>>(v, vx, vy, dx, pdx, dy, dt, TILEY/TIDSY);

   cutilCheckMsg("advectVelocity_k failed.");
}

extern "C"
   void diffuseProject(cData *vx, cData *vy, int dx, int dy, float dt, float visc) 
{ 
   // Forward FFT
   cufftExecR2C(planr2c, (cufftReal*)vx, (cufftComplex*)vx); 
   cufftExecR2C(planr2c, (cufftReal*)vy, (cufftComplex*)vy);

   uint3 grid = make_uint3((dx/TILEX)+(!(dx%TILEX)?0:1), 
      (dy/TILEY)+(!(dy%TILEY)?0:1), 1);
   uint3 tids = make_uint3(TIDSX, TIDSY, 1);

   diffuseProject_k<<<grid, tids>>>(vx, vy, dx, dy, dt, visc, TILEY/TIDSY);
   cutilCheckMsg("diffuseProject_k failed.");

   // Inverse FFT
   cufftExecC2R(planc2r, (cufftComplex*)vx, (cufftReal*)vx); 
   cufftExecC2R(planc2r, (cufftComplex*)vy, (cufftReal*)vy);
}

extern "C"
   void updateVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy) 
{ 
   dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));
   dim3 tids(TIDSX, TIDSY);

   updateVelocity_k<<<grid, tids>>>(v, vx, vy, dx, pdx, dy, TILEY/TIDSY, tPitch);
   cutilCheckMsg("updateVelocity_k failed.");
}

extern "C"
   void advectParticles(GLuint vbo, cData *v, int dx, int dy, float dt) 
{
   dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));
   dim3 tids(TIDSX, TIDSY);

   cData *p;
   cutilSafeCall(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
   size_t num_bytes; 
   cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&p, &num_bytes,  
      cuda_vbo_resource));
   cutilCheckMsg("cudaGraphicsResourceGetMappedPointer failed");

   advectParticles_k<<<grid, tids>>>(p, v, dx, dy, dt, TILEY/TIDSY, tPitch);
   cutilCheckMsg("advectParticles_k failed.");

   cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

   cutilCheckMsg("cudaGraphicsUnmapResources failed");
}
