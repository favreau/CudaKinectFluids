#pragma once

#include <windows.h>
#include <nuiapi.h>

struct Float3 
{
   float x,y,z;
};

class KinectWrapper
{

public:

   KinectWrapper(void);
   ~KinectWrapper(void);

public:

   void  initialize();
   BYTE* getDepthFrame();
   BYTE* getVideoFrame();
   bool  getSkeletonPositions(Float3* positions);
   
public:
   void  DepthToWorld(int x, int y, int depthValue, float& rx, float& ry, float& rz);
   float RawDepthToMeters(int depthValue);

   RGBQUAD KinNuiShortToQuadDepth( USHORT s );

private:

   int                m_mouse_x;
   int                m_mouse_y;

   HANDLE             m_skeletons;
   HANDLE             m_hNextDepthFrameEvent; 
   HANDLE             m_hNextVideoFrameEvent;
   HANDLE             m_hNextSkeletonEvent;
   HANDLE             m_pVideoStreamHandle;
   HANDLE             m_pDepthStreamHandle;
   NUI_SKELETON_FRAME m_skeletonFrame;
};

