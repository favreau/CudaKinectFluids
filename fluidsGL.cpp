/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/
// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_gl_inline.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h
#include <rendercheck_gl.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// CUDA FFT Libraries
#include <cufft.h>

// Shared Library Test Functions
#include <shrQATest.h>

// Kinect
#define REFRESH_DELAY	  10 //ms
#include "KinectWrapper.h"
KinectWrapper* kinectWrapper = NULL;
GLubyte* ubImage = NULL;

#ifdef WIN32
bool IsOpenGLAvailable(const char *appName) { return true; }
#else
#if (defined(__APPLE__) || defined(MACOSX))
bool IsOpenGLAvailable(const char *appName) { return true; }
#else
// check if this is a linux machine
#include <X11/Xlib.h>

bool IsOpenGLAvailable(const char *appName)
{
   Display *Xdisplay = XOpenDisplay(NULL);
   if (Xdisplay == NULL) {
      return false;
   } else {
      XCloseDisplay(Xdisplay);
      return true;
   }
}
#endif
#endif

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include "defines.h"
#include "fluidsGL_kernels.h"

#define MAX_EPSILON_ERROR 1.0f

const char *sSDKname = "fluidsGL";

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
   "fluidsGL.ppm",
   NULL
};

const char *sReference[] =
{
   "ref_fluidsGL.ppm",
   NULL
};

#define getmin(a,b) (a < b ? a : b)
#define getmax(a,b) (a > b ? a : b)

// CUDA example code that implements the frequency space version of 
// Jos Stam's paper 'Stable Fluids' in 2D. This application uses the 
// CUDA FFT library (CUFFT) to perform velocity diffusion and to 
// force non-divergence in the velocity field at each time step. It uses 
// CUDA-OpenGL interoperability to update the particle field directly
// instead of doing a copy to system memory before drawing. Texture is
// used for automatic bilinear interpolation at the velocity advection step. 

void cleanup(void);
void reshape(int x, int y);
#if 0
#else
void timerEvent(int value);
#endif // 0

// CUFFT plan handle
cufftHandle planr2c;
cufftHandle planc2r;
static cData *vxfield = NULL;
static cData *vyfield = NULL;

cData *hvfield = NULL;
cData *dvfield = NULL;
static int wWidth = DIMX;
static int wHeight = DIMY;

//static int clicked = 0;
static int fpsCount = 0;
static int fpsLimit = 1;
unsigned int timer;

// Particle data
GLuint vbo = 0;                 // OpenGL vertex buffer object
struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange
static cData *particles = NULL; // particle positions in host memory

static const int WindowsWidth  = 1024;
static const int WindowsHeight = 1024;
static int lastx = WindowsWidth/2, lasty = WindowsHeight/2, lastz = 0.f;

// Texture pitch
size_t tPitch = 0; // Now this is compatible with gcc in 64-bit

bool g_bQAReadback     = false;
bool g_bQAAddTestForce = true;
int  g_iFrameToCompare = 100;
int  g_TotalErrors     = 0;

// CheckFBO/BackBuffer class objects
CheckRender       *g_CheckRender = NULL;

extern "C" void addForces(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r);
extern "C" void advectVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy, float dt);
extern "C" void diffuseProject(cData *vx, cData *vy, int dx, int dy, float dt, float visc);
extern "C" void updateVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy);
extern "C" void advectParticles(GLuint vbo, cData *v, int dx, int dy, float dt);


void simulateFluids(void)
{
   // simulate fluid
   advectVelocity(dvfield, (float*)vxfield, (float*)vyfield, DIMX, RPADW, DIMY, DT);
   diffuseProject(vxfield, vyfield, CPADW, DIMX, DT, VIS);
   updateVelocity(dvfield, (float*)vxfield, (float*)vyfield, DIMX, RPADW, DIMY);
   advectParticles(vbo, dvfield, DIMX, DIMY, DT);
}

void TexFunc(void)
{
   glEnable(GL_TEXTURE_2D);
   glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
   glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
   glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

   glTexImage2D(GL_TEXTURE_2D, 0, 3, 640, 480, 0, GL_RGBA, GL_UNSIGNED_BYTE, ubImage);

   glBegin(GL_POLYGON);
   glTexCoord2f(1.0, 1.0);
   glVertex3f(0.5f-0.5,0.5+0.4, -0.1f);
   glTexCoord2f(0.0, 1.0);
   glVertex3f(0.5f+0.5,0.5+0.4, -0.1f);
   glTexCoord2f(0.0, 0.0);
   glVertex3f(0.5f+0.5,0.5-0.4, -0.1f);
   glTexCoord2f(1.0, 0.0);
   glVertex3f(0.5f-0.5,0.5-0.4, -0.1f);
   glEnd();

   glDisable(GL_TEXTURE_2D);
}

void display(void) 
{  
   if (!g_bQAReadback) {
      cutilCheckError(cutStartTimer(timer));  
      simulateFluids();
   }

   // render points from vertex buffer
   glClear(GL_COLOR_BUFFER_BIT);
   TexFunc();
   glColor4f(1.0f,1.0f,1.0f,1.0f); 
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glDisable(GL_CULL_FACE); 
   glEnable(GL_BLEND);

#if 0
   glEnable(GL_POINT_SMOOTH);
   glPointSize(3);
   glDisable(GL_DEPTH_TEST);
#endif // 0

   glEnableClientState(GL_VERTEX_ARRAY);    

   glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
   glVertexPointer(2, GL_FLOAT, 0, NULL);

   glDrawArrays(GL_POINTS, 0, DS);
   /*
   glPushMatrix();
   GLUquadricObj *sph1 = gluNewQuadric(); 
   glColor3f(255.f,0.0f,0.0f); // couleur de la sphère
   glTranslatef( lastx/1024.f, lasty/1024.f, lastz/1024.f );
   gluSphere(sph1,0.02,30,30); 
   glPopMatrix();
   */
   
   glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
   
   glDisableClientState(GL_VERTEX_ARRAY); 
   glDisableClientState(GL_TEXTURE_COORD_ARRAY); 
   glDisable(GL_TEXTURE_2D);

   if (g_bQAReadback) {
      return;
   }

   // Finish timing before swap buffers to avoid refresh sync
   cutilCheckError(cutStopTimer(timer));
   glutSwapBuffers();

   fpsCount++;
   if (fpsCount == fpsLimit) {
      char fps[256];
      float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
      sprintf(fps, "Cuda/GL Stable Fluids (%d x %d): %3.1f fps", DIMX, DIMY, ifps);  
      glutSetWindowTitle(fps);
      fpsCount = 0; 
      fpsLimit = (int)getmax(ifps, 1.f);
      cutilCheckError(cutResetTimer(timer));  
   }

   glutPostRedisplay();
}

// very simple von neumann middle-square prng.  can't use rand() in -qatest
// mode because its implementation varies across platforms which makes testing
// for consistency in the important parts of this program difficult.
float myrand(void)
{
   static int seed = 72191;
   char sq[22];

   if (g_bQAReadback) {
      seed *= seed;
      sprintf(sq, "%010d", seed);
      // pull the middle 5 digits out of sq
      sq[8] = 0;
      seed = atoi(&sq[3]);

      return seed/99999.f;
   } else {
      return rand()/(float)RAND_MAX;
   }
}

void initParticles(cData *p, int dx, int dy) 
{
   int i, j;
   for (i = 0; i < dy; i++) 
   {
      for (j = 0; j < dx; j++) 
      {
         p[i*dx+j].x = (j+0.5f+(myrand() - 0.5f))/dx;
         p[i*dx+j].y = (i+0.5f+(myrand() - 0.5f))/dy;
      }
   }
}

void keyboard( unsigned char key, int x, int y) 
{
   switch( key) {
   case 27:
      exit (0);
      break;
   case 'r':
      memset(hvfield, 0, sizeof(cData) * DS);
      cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS, 
         cudaMemcpyHostToDevice);

      initParticles(particles, DIMX, DIMY);

      cudaGraphicsUnregisterResource(cuda_vbo_resource);

      cutilCheckMsg("cudaGraphicsUnregisterBuffer failed");

      glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
      glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(cData) * DS, 
         particles, GL_DYNAMIC_DRAW_ARB);
      glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

      cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);

      cutilCheckMsg("cudaGraphicsGLRegisterBuffer failed");
      break;
   default: break;
   }
}

#if 0
void click(int button, int updown, int x, int y) 
{
   lastx = x; lasty = y;
   clicked = !clicked;
}
#endif //0

#if 0
void motion (int x, int y) 
#else
void timerEvent(int value)
#endif // 0
{
   Float3 positions[20];
   memset( positions, 0, 20*sizeof(Float3) );

   ubImage = kinectWrapper->getVideoFrame();
   kinectWrapper->getSkeletonPositions( positions );
   int dx = WindowsWidth/2;
   int dy = WindowsHeight/2;
   float x = dx-WindowsWidth*positions[NUI_SKELETON_POSITION_HAND_RIGHT].x;
   float y = dy-WindowsHeight*positions[NUI_SKELETON_POSITION_HAND_RIGHT].y;

   float r = positions[NUI_SKELETON_POSITION_HAND_RIGHT].z - positions[NUI_SKELETON_POSITION_HAND_LEFT].z;

   if( r < 0.f )
   {
      // Convert motion coordinates to domain
      float fx = (lastx / (float)wWidth);        
      float fy = (lasty / (float)wHeight);
      int nx = (int)(fx * DIMX);        
      int ny = (int)(fy * DIMY);   

      if (nx < DIMX-FR && nx > FR-1 && ny < DIMY-FR && ny > FR-1) 
      {
         int ddx = x - lastx;
         int ddy = y - lasty;
         fx = ddx / (float)wWidth;
         fy = ddy / (float)wHeight;
         int spy = ny-FR;
         int spx = nx-FR;
         addForces(dvfield, DIMX, DIMY, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);
         lastx = x; 
         lasty = y;
      } 
      else {
         lastx = WindowsWidth/2; 
         lasty = WindowsHeight/2;
      }
   }
   glutPostRedisplay();
#if 0
#else
   glutTimerFunc(REFRESH_DELAY, timerEvent,0);
#endif // 0
}

void reshape(int x, int y) {
   wWidth = x; wHeight = y;
   glViewport(0, 0, x, y);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(0, 1, 1, 0, 0, 1); 
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   glutPostRedisplay();
}

void cleanup(void) 
{
   delete kinectWrapper;
   delete ubImage;

   cudaGraphicsUnregisterResource(cuda_vbo_resource);

   unbindTexture();
   deleteTexture();

   // Free all host and device resources
   free(hvfield); free(particles); 
   cudaFree(dvfield); 
   cudaFree(vxfield); cudaFree(vyfield);
   cufftDestroy(planr2c);
   cufftDestroy(planc2r);

   glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
   glDeleteBuffersARB(1, &vbo);

   cutilCheckError(cutDeleteTimer(timer));  
}

int initGL(int *argc, char **argv)
{
   size_t len(640*480*4);
   ubImage = new GLubyte[len];
   memset( ubImage, 0, len ); 

   if (IsOpenGLAvailable(sSDKname)) {
      fprintf( stderr, "   OpenGL device is Available\n");
   } else {
      fprintf( stderr, "   OpenGL device is NOT Available, [%s] exiting...\n", sSDKname );
      shrQAFinishExit(*argc, (const char **)argv, QA_WAIVED);
      return CUTFalse;
   }

   glutInit(argc, argv);
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowSize(WindowsWidth, WindowsHeight);
   glutCreateWindow("Compute Stable Fluids");
   glutDisplayFunc(display);
   glutKeyboardFunc(keyboard);
#if 0
   glutMouseFunc(click);
   glutMotionFunc(motion);
#else
   // viewport
   glClearColor(0.0, 0.0, 0.0, 1.0);
   glDisable(GL_DEPTH_TEST);
   glViewport(0, 0, DIMX, DIMY);

   // projection
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   gluPerspective(60.0, (GLfloat)DIMX / (GLfloat) DIMY, 0.1, 10.0);

   // set view matrix
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();

   glutTimerFunc(REFRESH_DELAY, timerEvent,0);
   //glutFullScreen();
#endif // 0
   glutReshapeFunc(reshape);


   glewInit();
   if (! glewIsSupported(
      "GL_ARB_vertex_buffer_object"
      )) 
   {
      fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
      fflush( stderr);
      return CUTFalse;
   }
   return CUTTrue;
}


int main(int argc, char** argv) 
{
   int devID;
   cudaDeviceProp deviceProps;
   shrQAStart( argc, argv );
   printf("[%s] - [OpenGL/CUDA simulation] starting...\n", sSDKname);

   // First initialize OpenGL context, so we can properly set the GL for CUDA.
   // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
   if (CUTFalse == initGL(&argc, argv)) {
      shrQAFinishExit(argc, (const char **)argv, QA_WAIVED);
   }

   // use command-line specified CUDA device, otherwise use device with highest Gflops/s
   if (cutCheckCmdLineFlag(argc, (const char**)argv, "device")) {
      devID = cutilGLDeviceInit(argc, argv);
      if (devID < 0) {
         printf("no CUDA Capable device found, exiting...\n");
         shrQAFinishExit(argc, (const char **)argv, QA_WAIVED);
      }
   } else {
      devID = cutGetMaxGflopsDeviceId();
      cutilSafeCall(cudaGLSetGLDevice(devID));
   }

   // get number of SMs on this GPU
   cutilSafeCall(cudaGetDeviceProperties(&deviceProps, devID));
   printf("CUDA device [%s] has %d Multi-Processors\n", 
      deviceProps.name, deviceProps.multiProcessorCount);

   // automated build testing harness
   if (cutCheckCmdLineFlag(argc, (const char **)argv, "qatest") ||
      cutCheckCmdLineFlag(argc, (const char **)argv, "noprompt"))
   {
      g_bQAReadback = true;
   }

   // Allocate and initialize host data
   GLint bsize;

   cutilCheckError(cutCreateTimer(&timer));
   cutilCheckError(cutResetTimer(timer));  

   hvfield = (cData*)malloc(sizeof(cData) * DS);
   memset(hvfield, 0, sizeof(cData) * DS);

   // Allocate and initialize device data
   cudaMallocPitch((void**)&dvfield, &tPitch, sizeof(cData)*DIMX, DIMY);

   cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS, 
      cudaMemcpyHostToDevice); 
   // Temporary complex velocity field data     
   cudaMalloc((void**)&vxfield, sizeof(cData) * PDS);
   cudaMalloc((void**)&vyfield, sizeof(cData) * PDS);

   setupTexture(DIMX, DIMY);
   bindTexture();

   // Create particle array
   particles = (cData*)malloc(sizeof(cData) * DS);
   memset(particles, 0, sizeof(cData) * DS);   

   initParticles(particles, DIMX, DIMY); 

   // Create CUFFT transform plan configuration
   cufftPlan2d(&planr2c, DIMX, DIMY, CUFFT_R2C);
   cufftPlan2d(&planc2r, DIMX, DIMY, CUFFT_C2R);
   // TODO: update kernels to use the new unpadded memory layout for perf
   // rather than the old FFTW-compatible layout
   cufftSetCompatibilityMode(planr2c, CUFFT_COMPATIBILITY_FFTW_PADDING);
   cufftSetCompatibilityMode(planc2r, CUFFT_COMPATIBILITY_FFTW_PADDING);

   glGenBuffersARB(1, &vbo);
   glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
   glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(cData) * DS, 
      particles, GL_DYNAMIC_DRAW_ARB);

   glGetBufferParameterivARB(GL_ARRAY_BUFFER_ARB, GL_BUFFER_SIZE_ARB, &bsize); 
   if (bsize != (sizeof(cData) * DS))
      goto EXTERR;
   glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

   cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
   cutilCheckMsg("cudaGraphicsGLRegisterBuffer failed");

   // --------------------------------------------------------------------------------
   // Initialize Kinect
   // --------------------------------------------------------------------------------
   kinectWrapper = new KinectWrapper();
   kinectWrapper->initialize();
   // --------------------------------------------------------------------------------

   atexit(cleanup); 
   glutMainLoop();

   cutilDeviceReset();
   if (!g_bQAReadback) {
      shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
   }
   return 0;

EXTERR:
   printf("Failed to initialize GL extensions.\n");

   cutilDeviceReset();
   shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
}
