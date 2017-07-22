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

#pragma once

#include <windows.h>
#include <nuiapi.h>

struct Float3
{
    float x, y, z;
};

class KinectWrapper
{
public:
    KinectWrapper(void);
    ~KinectWrapper(void);

public:
    void initialize();
    BYTE *getDepthFrame();
    BYTE *getVideoFrame();
    bool getSkeletonPositions(Float3 *positions);

public:
    void DepthToWorld(int x, int y, int depthValue, float &rx, float &ry, float &rz);
    float RawDepthToMeters(int depthValue);

    RGBQUAD KinNuiShortToQuadDepth(USHORT s);

private:
    int m_mouse_x;
    int m_mouse_y;

    HANDLE m_skeletons;
    HANDLE m_hNextDepthFrameEvent;
    HANDLE m_hNextVideoFrameEvent;
    HANDLE m_hNextSkeletonEvent;
    HANDLE m_pVideoStreamHandle;
    HANDLE m_pDepthStreamHandle;
    NUI_SKELETON_FRAME m_skeletonFrame;
};
