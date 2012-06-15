@echo off
set INCLUDE=
set LIB=
set PATH=c:\windows;c:\windows\system32

SET VSINSTALLDIR="c:\Program Files\Microsoft Visual Studio 10.0"
SET VCINSTALLDIR="c:\Program Files\Microsoft Visual Studio 10.0\VC"
SET FrameworkDir32="c:\Windows\Microsoft.NET\Framework"
SET FrameworkVersion32=v4.0.30319
SET Framework35Version=v3.5

call "C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\vcvars32.bat"

:directX
	rem --------------------------------------------------------------------------------
	echo Microsoft DirectX
	rem --------------------------------------------------------------------------------
	set DIRECTX_HOME=C:\Program Files (x86)\Microsoft DirectX SDK (June 2010)
	set INCLUDE=%DIRECTX_HOME%\include;%INCLUDE%
	set LIB=%DIRECTX_HOME%\lib\x86;%LIB%

:nVidiaDirectXSDK
	rem --------------------------------------------------------------------------------
	echo nVidia Direct X SDK 11
	rem --------------------------------------------------------------------------------
	set NVIDIADIRECTXSDK_HOME=C:\local\nVidia\NVIDIA Direct3D SDK 11
	set INCLUDE=%NVIDIADIRECTXSDK_HOME%\include\DXUT10\Core;%INCLUDE%
	set INCLUDE=%NVIDIADIRECTXSDK_HOME%\include\DXUT10\Optional;%INCLUDE%
	set LIB=%NVIDIADIRECTXSDK_HOME%\lib;%LIB%
	
:etw_sdk
	rem --------------------------------------------------------------------------------
	echo Event Tracing for Windows
	rem --------------------------------------------------------------------------------
	set ETW_HOME=C:\Svn\Libraries\Logging\ETW\tags\01.00.00\x86
	set INCLUDE=%ETW_HOME%\include;%INCLUDE%
	set LIB=%ETW_HOME%\lib;%LIB%

:intel_sdk
	rem --------------------------------------------------------------------------------
	echo Intel OpenCL SDK
	rem --------------------------------------------------------------------------------
	set INTEL_HOME=C:\Program Files (x86)\Intel\OpenCL SDK\1.5
	set PATH=%PATH%;%INTEL_HOME%\bin\x86
	set INCLUDE=%INTEL_HOME%\include;%INCLUDE%
	set LIB=%INTEL_HOME%\lib\x86;%LIB%

:gpu_toolkit
	rem --------------------------------------------------------------------------------
	echo NVIDIA GPU Computing Toolkit
	rem --------------------------------------------------------------------------------
	set GPU_HOME=C:\local\nVidia\win32\NVIDIA GPU Computing Toolkit
	set PATH=%PATH%;%GPU_HOME%\CUDA\v4.1\bin
	set INCLUDE=%GPU_HOME%\CUDA\v4.1\include;%INCLUDE%
	set LIB=%GPU_HOME%\CUDA\v4.1\lib\Win32;%LIB%
	set PATH=%PATH%;%GPU_HOME%\CUDALibraries\bin\win32\Release
	set INCLUDE=%GPU_HOME%\CUDALibraries\common\inc;%INCLUDE%

:gpu_sdk
	rem --------------------------------------------------------------------------------
	echo NVIDIA GPU Computing SDK 4.1
	rem --------------------------------------------------------------------------------
	set GPU_HOME=C:\local\nVidia\win32\NVIDIA GPU Computing SDK 4.1

	echo   CUDA
	set PATH=%PATH%;%GPU_HOME%\C\bin\win32\release
	set PATH=%PATH%;%GPU_HOME%\C\common\bin
	set INCLUDE=%GPU_HOME%\C\common\inc;%INCLUDE%
	set LIB=%GPU_HOME%\C\common\lib\win32;%LIB%
	set NSIGHT_CUDA_DEBUGGER=1

	echo   OpenCL
	set INCLUDE=%GPU_HOME%\OpenCL\Common\inc;%INCLUDE%
	set LIB=%GPU_HOME%\OpenCL\Common\lib\win32;%LIB%

	echo   Shared
	set INCLUDE=%GPU_HOME%\shared\inc;%INCLUDE%
	set LIB=%GPU_HOME%\shared\lib\Win32;%LIB%

:nvapi
	rem --------------------------------------------------------------------------------
	echo  NVAPI
	rem --------------------------------------------------------------------------------
	set NVAPI_HOME=C:\local\nVidia\NVApi
	set INCLUDE=%NVAPI_HOME%;%INCLUDE%
	set LIB=%NVAPI_HOME%\x86;%LIB%

:kinect_sdk
	rem --------------------------------------------------------------------------------
	echo Kinect
	rem --------------------------------------------------------------------------------
	set KINECT_HOME=C:\Program Files\Microsoft SDKs\Kinect\v1.0 Beta2
	set PATH=%PATH%;%KINECT_HOME%
	set INCLUDE=%KINECT_HOME%\inc;%INCLUDE%
	set LIB=%KINECT_HOME%\lib\x86;%LIB%

devenv /useenv KinectFluids.sln
echo on
