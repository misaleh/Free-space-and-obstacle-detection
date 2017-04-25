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
#include "Occupancy.h"
using namespace cv;
using namespace std;


__global__ void polarGridKernel(ushort *d_Depth,unsigned char* d_test,float d_Q13, float d_Q23,int d_width,int d_height)
{
	 int j = (blockIdx.x * blockDim.x) + threadIdx.x; //width
	 int i = (blockIdx.y * blockDim.y) + threadIdx.y; //height
	 i = d_height - i -1;
	 float y,z;
	  if((i==0) || (j==0) || (i==d_height-1) || (j==d_width-1))
 		     return;    
		
	if( (d_Depth[d_width*i + j]	 ==  65535) || (d_Depth[d_width*i + j] ==  0) ) //ignore any unassigned depth
        		return;

           	y = (((float)(i)+d_Q13) * (float)d_Depth[d_width*i + j]) / d_Q23 ; //yc
            z = (float)d_Depth[d_width*i + j];//f*b/d
            //k = j;
            ///m = (z - zMinVal)/gridSize;
             //y dimension in point cloud is inverted
            //900 -> 600 , 1800 -> 800
            if(y <=  THRESHOLD) //what is good threshold  ??
            {
            	//PolarGrid.at<char>(m,k) = 0; //obstcale
            	d_test[d_width*i + j]= 0;
            }
            else //if(PolarGrid.at<char>(m,k) != 0)
            {
            	//PolarGrid.at<char>(m,k) = 255;
            	d_test[d_width*i + j]= 255;
	        }


}
void calculatePointCloudPolarGridGPU(const cv::Mat& Depth, cv::Mat &test, const cv::Mat& Q, int gridSize, double heightPercentage)
{
    // Getting the interesting parameters from Q, everything else is zero or one
    int k,m;
    float x,y,z;
    double zMinVal , zMaxVal; 
    float split = 1 - heightPercentage;
    float Q03 = Q.at<float>(0, 3);
    float Q13 = Q.at<float>(1, 3);
    float Q23 = Q.at<float>(2, 3);
    float Q32 = Q.at<float>(3, 2);
    float Q33 = Q.at<float>(3, 3);
    minMaxLoc( Depth, &zMinVal, &zMaxVal );
    int grid_rows = (zMaxVal - zMinVal )/gridSize  + 1;
	int grid_cols = Depth.cols ;
	test  = cv::Mat::zeros( (Depth.rows) , Depth.cols ,  CV_8UC1); //black means empty
	//Mat temp = Mat::zeros(  Depth.rows - (Depth.rows)*heightPercentage , Depth.cols ,  CV_8UC1);
 	//Mat PolarGrid;


    //PolarGrid = cv::Mat::zeros( grid_rows , grid_cols ,  CV_8U); //black means empty
    //PolarGrid = Scalar(127);
    test = Scalar(127);
    ushort *d_Depth;
    unsigned char * d_test;
    cudaMalloc((void**) &d_Depth, (Depth.rows)*(Depth.cols)*sizeof(ushort));
    cudaMalloc((void**) &d_test, (Depth.rows)*(Depth.cols)*sizeof(unsigned char));
    cudaMemcpy(d_Depth, Depth.data, (Depth.rows)*(Depth.cols)*sizeof(ushort), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_test, test.data, (Depth.rows)*(Depth.cols)*sizeof(unsigned char), cudaMemcpyHostToDevice); 

	dim3 threadsPerBlock(16,16,1);
	dim3 numBlocks( (Depth.cols)/16, (Depth.rows*heightPercentage)/16,1); 
	polarGridKernel<<<numBlocks , threadsPerBlock>>>(d_Depth,d_test,Q13,Q23,(Depth.cols),(Depth.rows));
	cudaMemcpy(test.data, d_test,(Depth.rows)*(Depth.cols)*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaFree(d_Depth);
	cudaFree(d_test);
 	//Mat L1((Depth.rows), (Depth.cols),CV_8UC1,tito);
 	//imshow("Toto",L1);
 	//waitKey(0);
}