#ifndef OCCUPANCY_H
#define OCCUPANCY_H
#define THRESHOLD  620 //the threshold to consider object and obstcale,(reversed)
#define all(v) ((v).begin()), ((v).end())
#define SIZE(v) ((int)((v).size()))
struct dimMinMax
{
float min_x;
float min_y;
float min_z;
float max_x;
float max_y;
float max_z;
};

/* CPU functions*/
//computes u disparity 
void computeVDisparity(cv::Mat &src,cv::Mat &dst); 
//computes v disparity 
void computeUDisparity(cv::Mat &src,cv::Mat &dst);
//computes cartesean grid from point cloud 
void computeCarteseanOccupancyMap(cv::Mat &src,cv::Mat &dst,const cv::Mat& Q,int gridSize,  dimMinMax & gridDims,double heightPercentage = 1);
//computes polar grid from point cloud 
void computePolarOccupancyMap(cv::Mat &src,cv::Mat &dst,cv::Mat &output,const cv::Mat& Q,int gridSize, dimMinMax & gridDims,int imrows);
//overloaded function returns dynamic array for each pixel and its original values, it is used to projectback from grid to image 
//with original quality
void computePolarOccupancyMap(cv::Mat &src,cv::Mat &dst,cv::Mat &output,std::vector< std::vector<int> >& Indecies,const cv::Mat& Q,int gridSize, dimMinMax & gridDims,int imrows);
//calculate point cloud from depth
void calculatePointCloud(const cv::Mat& depth, const cv::Mat& Q, cv::Mat& out3D, dimMinMax & gridDims,double heightPercentage = 1 );
//segement using thresholding on  the image 
void segmentThresholdImg(cv::Mat &mask,double heightPercentage = 1); 	
//saves point cloud to xyz file
void save(const cv::Mat& image3D, const std::string& fileName);
//convert polar grid back to image, using the grid only
void projectPolarToImage(cv::Mat &grid,cv::Mat &dst, dimMinMax & gridDims,const cv::Mat& Q, int gridSize);
//overloaded function used with the overloaded polar grid function
void projectPolarToImage(cv::Mat &grid,cv::Mat &dst,std::vector< std::vector<int> >& Indecies,double heightPercentage = 1);
// preform segementation using dynamic programming on the grid
void segementDP(cv::Mat &grid,cv::Mat &gridout); 
//segmentaion using thersholding on the paper
void segmentThreshold(cv::Mat &grid,cv::Mat &gridout); 	


/*GPU functions*/
//compute point cloud then polar grid on GPU
void calculatePointCloudPolarGridGPU(const cv::Mat& depth, cv::Mat &dst, const cv::Mat& Q, int gridSize,double heightPercentage = 1 );

#endif