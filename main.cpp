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

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <rendercheck_gl.h>

// CUDA standard includes
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// CUDA helper functions
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA FFT Libraries
#include <cufft.h>

// Kinect
#define REFRESH_DELAY 20 // ms
#include "KinectWrapper.h"
KinectWrapper *kinectWrapper = nullptr;
BYTE *ubImage = nullptr;
BYTE *ubDepth = nullptr;

// Param
float gParam = 110.f;
float gPointSize = 1.f;
float gTimer = 0.f;

// mouse controls
float2 gMousePosition;
float2 gMousePositionOld;
int mouse_buttons = 0;

#ifdef WIN32
bool IsOpenGLAvailable(const char *appName)
{
    return true;
}
#else
#if (defined(__APPLE__) || defined(MACOSX))
bool IsOpenGLAvailable(const char *appName)
{
    return true;
}
#else
// check if this is a linux machine
#include <X11/Xlib.h>

bool IsOpenGLAvailable(const char *appName)
{
    Display *Xdisplay = XOpenDisplay(nullptr);
    if (Xdisplay == nullptr)
    {
        return false;
    }
    else
    {
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
const char *sOriginal[] = {"fluidsGL.ppm", nullptr};

const char *sReference[] = {"ref_fluidsGL.ppm", nullptr};

#define getmin(a, b) (a < b ? a : b)
#define getmax(a, b) (a > b ? a : b)

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
void motion(int x, int y);
void mouse(int button, int state, int x, int y);

// CUFFT plan handle
cufftHandle planr2c;
cufftHandle planc2r;
static cData *vxfield = nullptr;
static cData *vyfield = nullptr;

cData *hvfield = nullptr;
cData *dvfield = nullptr;
static int wWidth = DIMX;
static int wHeight = DIMY;

// static int clicked = 0;
static int fpsCount = 0;
static int fpsLimit = 1;
unsigned int timer;

// Particle data
GLuint vbo = 0;                                       // OpenGL vertex buffer object
struct cudaGraphicsResource *cuda_vbo_resource;       // handles OpenGL-CUDA exchange
GLuint vbo_color = 0;                                 // OpenGL vertex buffer object
struct cudaGraphicsResource *cuda_vbo_color_resource; // handles OpenGL-CUDA exchange
static cData *particles = nullptr;                    // particle positions in host memory
static GLfloat *particuleColors = nullptr;

// Bitmap
unsigned char *bitmapImage = nullptr; // store image data

static const int WindowsWidth = 1024;
static const int WindowsHeight = WindowsWidth * 9 / 16;
static int lastx = WindowsWidth / 2, lasty = WindowsHeight / 2, lastz = 0;

// Texture pitch
size_t tPitch = 0; // Now this is compatible with gcc in 64-bit

bool g_bQAReadback = false;
bool g_bQAAddTestForce = true;
int g_iFrameToCompare = 100;
int g_TotalErrors = 0;

// CheckFBO/BackBuffer class objects
CheckRender *g_CheckRender = nullptr;

extern "C" void feelTheAttraction(cData *v, int dimx, int dimy, float3 position, float timer, float param);
extern "C" void initialize_scene();
extern "C" void h2d_kinect(BYTE *kinectVideo, BYTE *kinectDepth);
extern "C" void finalize_scene();

void initParticlesFromTexture(cData *p, int dx, int dy, const std::string &filename)
{
    FILE *filePtr(0);                  // our file pointer
    BITMAPFILEHEADER bitmapFileHeader; // our bitmap file header
    BITMAPINFOHEADER bitmapInfoHeader;
    DWORD imageIdx = 0; // image index counter

    if (bitmapImage == nullptr)
    {
        // open filename in read binary mode
        fopen_s(&filePtr, filename.c_str(), "rb");
        if (filePtr == nullptr)
            return;

        // read the bitmap file header
        fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);

        // verify that this is a bmp file by check bitmap id
        if (bitmapFileHeader.bfType != 0x4D42)
        {
            fclose(filePtr);
            return;
        }

        // read the bitmap info header
        fread(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);

        // move file point to the begging of bitmap data
        fseek(filePtr, bitmapFileHeader.bfOffBits, SEEK_SET);

        // allocate enough memory for the bitmap image data
        bitmapImage = new unsigned char[bitmapInfoHeader.biSizeImage];

        particuleColors = new float[bitmapInfoHeader.biSizeImage];

        // verify memory allocation
        if (!bitmapImage)
        {
            delete bitmapImage;
            fclose(filePtr);
            return;
        }

        // read in the bitmap image data
        fread(bitmapImage, bitmapInfoHeader.biSizeImage, 1, filePtr);

        // make sure bitmap image data was read
        if (bitmapImage == nullptr)
        {
            fclose(filePtr);
            return;
        }

        /*
        //swap the r and b values to get RGB (bitmap is BGR)
        for (imageIdx = 0; imageIdx < bitmapInfoHeader.biSizeImage; imageIdx += 3)
        {
           tempRGB = bitmapImage[imageIdx];
           bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
           bitmapImage[imageIdx + 2] = tempRGB;
        }
        */

        // close file and return bitmap image data
        fclose(filePtr);
    }

    int idx = 0; // bitmapInfoHeader.biSizeImage;
    int i, j;
    for (i = 0; i < dy; i++)
    {
        for (j = 0; j < dx; j++)
        {
            unsigned char r = bitmapImage[idx + 2];
            unsigned char g = bitmapImage[idx + 1];
            unsigned char b = bitmapImage[idx + 0];

            float R = r / 256.f;
            float G = g / 256.f;
            float B = b / 256.f;
            particuleColors[idx + 0] = R;
            particuleColors[idx + 1] = G;
            particuleColors[idx + 2] = B;

            p[i * dx + j].x = (j + 0.5f) / dx;
            p[i * dx + j].y = (i + 0.5f) / dy;
            idx += 3;
        }
    }
}

#if 0
void simulateFluids(void)
{
   // simulate fluid
   advectVelocity(dvfield, (float*)vxfield, (float*)vyfield, DIMX, RPADW, DIMY, DT);
   diffuseProject(vxfield, vyfield, CPADW, DIMX, DT, VIS);
   updateVelocity(dvfield, (float*)vxfield, (float*)vyfield, DIMX, RPADW, DIMY);
   advectParticles(vbo, dvfield, DIMX, DIMY, DT);
}
#endif // 0

void TexFunc(void)
{
    glEnable(GL_TEXTURE_2D);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

    glTexImage2D(GL_TEXTURE_2D, 0, 3, 640, 480, 0, GL_RGBA, GL_UNSIGNED_BYTE, ubImage);
    glBegin(GL_POLYGON);

    float dx = 2.f * 0.064f;
    float dy = 2.f * 0.048f;
    float dz = 0.f;
    glBegin(GL_POLYGON);
    glTexCoord2f(1.f, 1.f);
    glVertex3f(0.f, dy, dz);
    glTexCoord2f(0.0, 1.f);
    glVertex3f(dx, dy, dz);
    glTexCoord2f(0.f, 0.f);
    glVertex3f(dx, 0.f, dz);
    glTexCoord2f(1.f, 0.f);
    glVertex3f(0.f, 0.f, dz);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

void display(void)
{
    // render points from vertex buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if 1
    // Draw particles
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    glColorPointer(3, GL_FLOAT, 0, particuleColors);

    glPointSize(gPointSize);
    glDrawArrays(GL_POINTS, 0, DS);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
#endif // 0

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);

    // Finish timing before swap buffers to avoid refresh sync
    glutSwapBuffers();

    glutPostRedisplay();
}

// very simple von neumann middle-square prng.  can't use rand() in -qatest
// mode because its implementation varies across platforms which makes testing
// for consistency in the important parts of this program difficult.
float myrand(void)
{
    return rand() / (float)RAND_MAX;
}

void initParticles(cData *p, int dx, int dy)
{
    int i, j;
    for (i = 0; i < dy; i++)
    {
        for (j = 0; j < dx; j++)
        {
            p[i * dx + j].x = (j + 0.5f + (myrand() - 0.5f)) / dx;
            p[i * dx + j].y = (i + 0.5f + (myrand() - 0.5f)) / dy;
        }
    }
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 27:
        exit(0);
        break;
    case '+':
        gParam += 5.f;
        printf("%f\n", gParam);
        break;
    case '-':
        gParam -= 5.f;
        printf("%f\n", gParam);
        break;
    case 'p':
        gPointSize += 1.f;
        if (gPointSize > 6.f)
            gPointSize = 1.f;
        break;
    case 'r':
        memset(hvfield, 0, sizeof(cData) * DS);
        cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS, cudaMemcpyHostToDevice);

        // initParticles(particles, DIMX, DIMY);
        initParticlesFromTexture(particles, DIMX, DIMY, "./medias/021.bmp");

        cudaGraphicsUnregisterResource(cuda_vbo_resource);
        cudaGraphicsUnregisterResource(cuda_vbo_color_resource);

        // cutilCheckMsg("cudaGraphicsUnregisterBuffer failed");

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
        glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(cData) * DS, particles, GL_DYNAMIC_DRAW_ARB);
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo_color);
        glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(GLfloat) * 3 * DS, particuleColors, GL_DYNAMIC_DRAW_ARB);
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

        cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);
        cudaGraphicsGLRegisterBuffer(&cuda_vbo_color_resource, vbo_color, cudaGraphicsMapFlagsNone);

        // cutilCheckMsg("cudaGraphicsGLRegisterBuffer failed");
        break;
    default:
        break;
    }
}

void timerEvent(int value)
{
    Float3 positions[20];
    memset(positions, 0, 20 * sizeof(Float3));

    if (kinectWrapper)
    {
        ubImage = kinectWrapper->getVideoFrame();
        ubDepth = kinectWrapper->getDepthFrame();
        kinectWrapper->getSkeletonPositions(positions);
    }

#if 0
   int dx = WindowsWidth/2;
   int dy = WindowsHeight/2;
   
   // Cursor position
   float x = (positions[NUI_SKELETON_POSITION_HAND_RIGHT].x );
   float y = (positions[NUI_SKELETON_POSITION_HAND_RIGHT].y );

   //x *= 1000.f;
   //y *= 1000.f;

   //float r = positions[NUI_SKELETON_POSITION_HAND_RIGHT].z - positions[NUI_SKELETON_POSITION_HAND_LEFT].z;
   //if( r < 0.f )
   {
      // Convert motion coordinates to domain
      float fx = x;//(x / (float)wWidth);        
      float fy = y;//(y / (float)wHeight);
      int nx = (int)(fx * DIMX);        
      int ny = (int)(fy * DIMY);   

      if (nx < DIMX-FR && nx > FR-1 && ny < DIMY-FR && ny > FR-1) 
      {
         int ddx = dx + (lastx-x);
         int ddy = dy + (lasty-y);
         fx = ddx / (float)wWidth;
         fy = ddy / (float)wHeight;
         int spy = ny-FR;
         int spx = nx-FR;
         addForces(dvfield, DIMX, DIMY, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);
         lastx = x;
         lasty = y;
      } 
      else {
         lastx = 0.f; 
         lasty = 0.f;
      }
   }
#else
    float3 kinectPosition;
    kinectPosition.x = (positions[NUI_SKELETON_POSITION_HEAD].x + positions[NUI_SKELETON_POSITION_HAND_LEFT].x +
                        positions[NUI_SKELETON_POSITION_HAND_RIGHT].x) /
                       3.f;
    kinectPosition.y = -(positions[NUI_SKELETON_POSITION_HEAD].y + positions[NUI_SKELETON_POSITION_HAND_LEFT].y +
                         positions[NUI_SKELETON_POSITION_HAND_RIGHT].y) /
                       3.f;

    kinectPosition.x = gMousePosition.x / WindowsWidth;
    kinectPosition.y = gMousePosition.y / WindowsHeight;
    kinectPosition.z = 0.f;

    h2d_kinect(ubImage, ubDepth);
    feelTheAttraction(dvfield, DIMX, DIMY, kinectPosition, gTimer, gParam /*+20*cos(gTimer)*/);

    gTimer += 0.1f;
#endif // 0
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void reshape(int x, int y)
{
    wWidth = x;
    wHeight = y;
    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 1, 0, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutPostRedisplay();
}

// Mouse event handlers
//*****************************************************************************
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1 << button;
        gMousePosition.x = x - WindowsWidth / 2.f;
        gMousePosition.y = y - WindowsHeight / 2.f;

        gMousePositionOld = gMousePosition;
    }
    else
    {
        if (state == GLUT_UP)
        {
            mouse_buttons = 0;
            gMousePosition.x = 0;
            gMousePosition.y = 0;
            gMousePositionOld = gMousePosition;
        }
    }
}

void motion(int x, int y)
{
    switch (mouse_buttons)
    {
    case 1:
        gMousePositionOld = gMousePosition;
        gMousePosition.x = x - WindowsWidth / 2.f;
        gMousePosition.y = y - WindowsHeight / 2.f;
        break;
    case 2:
        break;
    case 4:
        break;
    default:
        gMousePosition.x = 0;
        gMousePosition.y = 0;
        gMousePositionOld = gMousePosition;
    }
}

void cleanup(void)
{
    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    cudaGraphicsUnregisterResource(cuda_vbo_color_resource);

    unbindTexture();
    deleteTexture();

    // Free all host and device resources
    free(hvfield);
    free(particles);
    cudaFree(dvfield);
    cudaFree(vxfield);
    cudaFree(vyfield);
    cufftDestroy(planr2c);
    cufftDestroy(planc2r);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    glDeleteBuffersARB(1, &vbo);
    glDeleteBuffersARB(1, &vbo_color);

    delete kinectWrapper;
    kinectWrapper = nullptr;

    finalize_scene();

    delete bitmapImage;
    delete particuleColors;
}

int initGL(int *argc, char **argv)
{
    // Kinect Image
    size_t len(640 * 480 * 4);
    ubImage = new GLubyte[len];

    // Kinect Image
    len = 320 * 240 * 2;
    ubDepth = new GLubyte[len];

    if (IsOpenGLAvailable(sSDKname))
    {
        fprintf(stderr, "   OpenGL device is Available\n");
    }
    else
    {
        fprintf(stderr, "   OpenGL device is NOT Available, [%s] exiting...\n", sSDKname);
        return false;
    }

    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(WindowsWidth, WindowsHeight);
    glutCreateWindow("Compute Stable Fluids");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);

    // glutMouseFunc(click);

    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    // glutFullScreen();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    glutReshapeFunc(reshape);

    glewInit();
    if (!glewIsSupported("GL_ARB_vertex_buffer_object"))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    int devID;
    cudaDeviceProp deviceProps;
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA
    // interop.
    initGL(&argc, argv);

    // use command-line specified CUDA device, otherwise use device with highest
    // Gflops/s
    devID = findCudaGLDevice(argc, const_cast<const char **>(argv));

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name, deviceProps.multiProcessorCount);

    // Allocate and initialize host data
    GLint bsize;

    // Init Cuda
    initialize_scene();

    hvfield = (cData *)malloc(sizeof(cData) * DS);
    memset(hvfield, 0, sizeof(cData) * DS);

    // Allocate and initialize device data
    cudaMallocPitch((void **)&dvfield, &tPitch, sizeof(cData) * DIMX, DIMY);

    cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS, cudaMemcpyHostToDevice);
    // Temporary complex velocity field data
    cudaMalloc((void **)&vxfield, sizeof(cData) * PDS);
    cudaMalloc((void **)&vyfield, sizeof(cData) * PDS);

    setupTexture(DIMX, DIMY);
    bindTexture();

    // Create particle array
    particles = (cData *)malloc(sizeof(cData) * DS);
    memset(particles, 0, sizeof(cData) * DS);

    // Create color array
    particles = (cData *)malloc(sizeof(cData) * DS);
    memset(particles, 0, sizeof(cData) * DS);

    // initParticles(particles, DIMX, DIMY);
    initParticlesFromTexture(particles, DIMX, DIMY, "./medias/023.bmp");

    // Create CUFFT transform plan configuration
    cufftPlan2d(&planr2c, DIMX, DIMY, CUFFT_R2C);
    cufftPlan2d(&planc2r, DIMX, DIMY, CUFFT_C2R);
    // TODO: update kernels to use the new unpadded memory layout for perf
    // rather than the old FFTW-compatible layout
    cufftSetCompatibilityMode(planr2c, CUFFT_COMPATIBILITY_FFTW_PADDING);
    cufftSetCompatibilityMode(planc2r, CUFFT_COMPATIBILITY_FFTW_PADDING);

    glGenBuffersARB(1, &vbo);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(cData) * DS, particles, GL_DYNAMIC_DRAW_ARB);
    glGetBufferParameterivARB(GL_ARRAY_BUFFER_ARB, GL_BUFFER_SIZE_ARB, &bsize);
    if (bsize != (sizeof(cData) * DS))
        goto EXTERR;
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    glGenBuffersARB(1, &vbo_color);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo_color);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(GLfloat) * 3 * DS, particuleColors, GL_DYNAMIC_DRAW_ARB);
    glGetBufferParameterivARB(GL_ARRAY_BUFFER_ARB, GL_BUFFER_SIZE_ARB, &bsize);
    if (bsize != (sizeof(GLfloat) * 3 * DS))
        goto EXTERR;
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_color_resource, vbo_color, cudaGraphicsMapFlagsNone));
    // cutilCheckMsg("cudaGraphicsGLRegisterBuffer failed");

    // --------------------------------------------------------------------------------
    // Initialize Kinect
    // --------------------------------------------------------------------------------
    kinectWrapper = new KinectWrapper();
    kinectWrapper->initialize();
    // --------------------------------------------------------------------------------

    atexit(cleanup);
    glutMainLoop();

    cudaDeviceReset();
    return 0;

EXTERR:
    printf("Failed to initialize GL extensions.\n");

    cudaDeviceReset();
}
