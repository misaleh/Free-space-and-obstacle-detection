### Free Space and obstacle detection using occupancy grids
**Tools:** OpenCV 2.4, C++, CUDA. 

This impelementaion is based on [Free Space Computation Using Stochastic Occupancy Grids and Dynamic Programming](http://vision.jhu.edu/iccv2007-wdv/WDV07-badino.pdf)
with some modifications.

The code is tested with Zed stereo camera and nvidia jetson tk1, and the parameters depend on the camera(calibraion) and the position of the camera.

This repo is still under development.

To make cpu and gpu code

type `make` then `make run`

To make CPU only, remove `#define GPU` from `main.cpp`

type `make cpu` then `make run_cpu`

To make GPU only, remove `#define CPU` from `main.cpp`

type `make gpu` then `make run_gpu`


**Input image and its Depth**

<img src="input/Left.png"  width="400">
<img src="input/Depth.png"  width="400">

**Polar Occypancy Grid with different sizes**


<img src="output/polargrid.png"  width="400">
<img src="output/polargrid1.png" width="400">


**Cartesian Occypancy Grid with different sizes**


<img src="output/grid.png" width="400" >
<img src="output/grid1.png" width="400" >

**Segmentation of polar grid** 


**Thresholding**

<img src="output/gridTh.png" width="400" >

**Free space**

<img src="output/projectedImg.png" width="400" >

**Dynamic programming**

<img src="output/gridDP.png" width="400" >

**Free space**

<img src="output/projectedImgDP.png" width="400" >

**GPU Implementation**

<img src="output/FreespaceGPU.png" width="400" >

