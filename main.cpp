// TODO
// dynamic(make grey ok ??) -> project back(dyamic array(array of grid size each element is vector of Vs))->inclines 
//1. improve grid
  //may use u-disparity instead of point cloud to accelerate the computation (but need to know the ground plane)
  //http://emotion.inrialpes.fr/people/spalanzani/Doc/Publication/occGridInUDisp-IROS2010.pdf
  //point cloud can be used from normals to estimate planes(good for path planning)
  //http://www.araa.asn.au/acra/acra2010/papers/pap151s1-file1.pdf
//2.use ego motion
  //Cost function 
  //Filter map using binary bayes filter(common log odds)(can work for 360 only check Dempster-Shafer) (will need propabilty)
  //Zed covariance https://github.com/stereolabs/zed-ros-wrapper/issues/78
  // sigma(mm) =0.25*(z*z)/(41.0316+z)
  //probabilty of filled is based on the number of points  

//3. More testing
//4. use information from image 
//5. find a way to work with inclines
//6. improve GPU code
//7 .clean code and optimization
/*
The code computes freespace and obstcales from stereo images and its depth
It also calculate the point cloud, polar occupancy grid and cartesean  occupancy grid
There is an alternative GPU function, but it still need some modifications 
use #define GPU to run GPU code or #define CPU to run CPU code, use can define both to run both of them
#define WRITE_CLOUD to write the point cloud file, not recommended if not needed as it consumes a lot of time 

Parameters for this code are
DEPTH_RATIO, used for background subtraction
THRESHOLD  the height threshold to be considerd as obstcale 
And the Q matrix, used for point cloud calculation 
*/
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/core/gpumat.hpp"
#include <iostream>
#include <string>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include "Occupancy.h"
#define CELL_SIZE 20 // each cell in grid CELL_SIZE*CELL_SIZE mm
#define DEPTH_RATIO 0.38  // ratio of depth map from the ground to calculate the grid, used for background subtraction
using namespace cv;
using namespace std;

#define CPU  // or GPU 
#define GPU
#define WRITE_CLOUD  // to write the point cloud file 


int main()
{
  // http://answers.ros.org/question/53654/how-to-save-depth-image-loss-less-to-disk/
  // USE -1 FLAG
  // images from zed are 16UC1
  Mat Depth = imread("input/Depth.png", -1);
  Mat imgLeft = imread("input/Left.png");
  Mat grid2D, polarGrid2D, outputGPU, output, visual, visualGPU, visualTh,visualDP,threshold,
    image3D,gridSegm,gridTh; // 2D occupancy grid , intialize grey(unknown)
  Mat projectedImg = Mat( Depth.rows, Depth.cols, CV_8UC1);
  projectedImg = Scalar(127);  
  Mat projectedImgDP = Mat( Depth.rows, Depth.cols, CV_8UC1);
  projectedImgDP = Scalar(127);

  vector< vector<int> > Indecies;
  dimMinMax gridDims;
  double min, max;

  Matx44d Q_VGA = Matx44d(/*ZED VGA Q Matrix*/
                          1.0, 0.0, 0.0, -336.99, 0.0, 1.0, 0.0, -186.356, 0.0,
                          0.0, 0.0, 349.547, 0.0, 0.0, -1.0 / 120, 0);
  float Q_array[4][4] = { /*Original ZED HD Q Matrix */
                          { 1.0, 0.0, 0.0, -642.979 },
                          { 0.0, 1.0, 0.0, -357.712 },
                          { 0.0, 0.0, 0.0, 699.094 },
                          { 0.0, 0.0, -1.0 / 120, 0 }
  };
  float Q_array2[4][4] = { /*ZED HD Q Matrix, tuning Cy to compensate for inclination */
                          { 1.0, 0.0, 0.0, -642.979 },
                          { 0.0, 1.0, 0.0, -413.712 },
                          { 0.0, 0.0, 0.0, 699.094 },
                          { 0.0, 0.0, -1.0 / 120, 0 }
                        };
  Mat Q = Mat(4, 4, CV_32F, Q_array2); /*ZED HD Q Matrix as Opencv Mat */

  /************** CPU CODE ***************/
#ifdef CPU
  calculatePointCloud(
    Depth, Q, image3D, gridDims,
    DEPTH_RATIO); // calculate point cloud from depth (much faster than ZED)
  /* print main and max vlaue in pointcloud for each axis*/
  cerr << "max_z " << gridDims.max_z << " "
       << "min_z " << gridDims.min_z << endl;
  cerr << "max_x " << gridDims.max_x << " "
       << "min_x " << gridDims.min_x << endl;
  cerr << "max_y " << gridDims.max_y << " "
       << "min_y " << gridDims.min_y << endl;
#ifdef WRITE_CLOUD     
  save(image3D, "pointcloud.xyz"); // save point cloud
#endif
  computeCarteseanOccupancyMap(
    image3D, grid2D, Q, CELL_SIZE, gridDims,
    DEPTH_RATIO); // from point cloud to 2D  Cartesean occupancy grid
  computePolarOccupancyMap(
    image3D, polarGrid2D, output, Q, CELL_SIZE, gridDims,
    Depth.rows); // from point cloud to 2D Polar occupancy grid, also return the free space(output)
  //computePolarOccupancyMap(
   // image3D, polarGrid2D, output,Indecies, Q, CELL_SIZE, gridDims,
   // Depth.rows);
  segmentThresholdImg(output, DEPTH_RATIO);// preform segmentaion by thresholding on the image it self
  segementDP(polarGrid2D,gridSegm);
  segmentThreshold(polarGrid2D,gridTh);
  /*project back from grid to image, not used now */
  //projectPolarToImage(gridTh,projectedImg,Indecies,DEPTH_RATIO);
 // projectPolarToImage(gridSegm,projectedImgDP,Indecies,DEPTH_RATIO);
  projectPolarToImage(gridSegm,projectedImgDP, gridDims,Q,CELL_SIZE );
  projectPolarToImage(gridTh,projectedImg, gridDims,Q,CELL_SIZE );

#endif
  /*************** GPU CODE *******************/
#ifdef GPU
  /*this function assumes the grid size = 1 and and calculate the free space directly 
    and the point cloud implicitly for optimization in GPU, it can be edited for other sizes 
  */
  calculatePointCloudPolarGridGPU(Depth, outputGPU, Q, CELL_SIZE, DEPTH_RATIO);
#endif


/**************** Visualization *************************/
#ifdef GPU
  applyColorMap(outputGPU, outputGPU, COLORMAP_JET); // to make it colored
  addWeighted(imgLeft, 0.7, outputGPU, 0.3, 0.0,
              visualGPU); // mix original image with output
#endif
#ifdef CPU
  addWeighted(imgLeft, 0.7, output, 0.3, 0.0,
              visual); // mix original image with output
  applyColorMap(projectedImgDP, projectedImgDP, COLORMAP_JET); // to make it colored
  addWeighted(imgLeft, 0.7, projectedImgDP, 0.3, 0.0,
              visualDP); // mix original image with output
  applyColorMap(projectedImg, projectedImg, COLORMAP_JET); // to make it colored
  addWeighted(imgLeft, 0.7, projectedImg, 0.3, 0.0,
              visualTh); // mix original image with output
#endif
/************************** WRITING To Disk***************/
#ifdef CPU
  imwrite("output/Freespace.png", visual); // free space calculated from CPU CODE
  imwrite("output/grid.png", grid2D);
  imwrite("output/polargrid.png", polarGrid2D);
  imwrite("output/gridDP.png", gridSegm);
  imwrite("output/gridTh.png", gridTh);
  imwrite("output/projectedImg.png", visualTh);
  imwrite("output/projectedImgDP.png", visualDP);
#endif
#ifdef GPU
  imwrite("output/FreespaceGPU.png", visualGPU); // free space calculated from GPU CODE
#endif


  return 0;
}
