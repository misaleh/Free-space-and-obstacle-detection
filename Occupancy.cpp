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
using namespace cv;
using namespace std;
/*calculate U an V disparties (not needed for this impelementaion of occupancy grids)*/
void computeVDisparity(Mat &src,Mat &dst)
{
	unsigned char d;
	Mat tmpDst = Mat::zeros(dst.rows,dst.cols,CV_16U);

	for(int i = 0 ; i < src.rows ;  i++)
	{
		for(int j = 0; j < src.cols; j++)
		{
			d = src.at<char>(i,j);
			tmpDst.at<ushort>(i,d) = tmpDst.at<ushort>(i,d) +  1 ; 
		}
	}
	normalize(tmpDst, dst, 0, 255, NORM_MINMAX, CV_8U);//for visulaiztion 

}
void computeUDisparity(Mat &src,Mat &dst)
{
	unsigned char d; 
	Mat tmpDst = Mat::zeros(dst.rows,dst.cols,CV_16U);

	for(int i = 0 ; i < src.rows ;  i++)
	{
		for(int j = 0; j < src.cols; j++)
		{

			d = src.at<char>(i,j);
		//	printf("i = %d j=  %d  d= %d\n",i,j,d);
		//	printf("disp before %d\n",dst.at<char>(d,j));
			tmpDst.at<ushort>(d,j) = tmpDst.at<ushort>(d,j) +  1 ;  
		//	printf("disp after %d\n",dst.at<char>(d,j));
		}
	}
	normalize(tmpDst, dst, 0, 255, NORM_MINMAX, CV_8U);//for visulaiztion 
}


//zc = f*b/d
//xc = zc*(x-x0)/f
//yc = zc*(y-y0)/f
//heightPercentage: percentage of image 
//calculate point cloud from depth image
void calculatePointCloud(const cv::Mat& depth, const cv::Mat& Q, cv::Mat& out3D, struct dimMinMax & gridDims , double heightPercentage )
{
    // 3-channel matrix for containing the reprojected 3D world coordinates
    out3D = cv::Mat::zeros(depth.rows*heightPercentage ,depth.cols, CV_32FC3);
    gridDims.max_z = 0 ;
    gridDims.min_z = 65536;
    gridDims.max_x = -65536 ;
    gridDims.min_x = 65536;
    gridDims.max_y = -65536 ;
    gridDims.min_y = 65536;

    // Getting the interesting parameters from Q, everything else is zero or one
    float split = 1 - heightPercentage;
    float Q03 = Q.at<float>(0, 3);
    float Q13 = Q.at<float>(1, 3);
    float Q23 = Q.at<float>(2, 3);
    float Q32 = Q.at<float>(3, 2);
    float Q33 = Q.at<float>(3, 3);
    for (int i = (depth.rows)* split; i <(depth.rows) ; i++)
    {
        cv::Vec3f* out3D_ptr = out3D.ptr<cv::Vec3f>(i-(depth.rows)* split); //should be  -(depth.rows)* split
        for (int j = 1; j < depth.cols; j++)
        {
        	if( (depth.at<ushort>(i,j) ==  65535) || (depth.at<ushort>(i,j) ==  0) ) //ignore any unassigned depth
        		continue;

            cv::Vec3f& point = out3D_ptr[j];
            point[0] = ((static_cast<float>(j)+Q03) * (float)depth.at<ushort>(i,j)) / Q23;  //xc
            point[1] = ((static_cast<float>(i)+Q13) * (float)depth.at<ushort>(i,j)) / Q23 ; //yc
            point[2] = (float)depth.at<ushort>(i,j);//f*b/d
            if( point[2] < gridDims.min_z )
            	gridDims.min_z =  point[2];
            if( point[2] > gridDims.max_z )
            	gridDims.max_z = point[2];
            if( point[0] < gridDims.min_x )
            	gridDims.min_x =  point[0];
            if( point[0] > gridDims.max_x )
            	gridDims.max_x = point[0];
            if( point[1] < gridDims.min_y )
            	gridDims.min_y =  point[1];
            if( point[1] > gridDims.max_y )
            	gridDims.max_y = point[1];
            
        }
    }
}
//source for function save :https://github.com/bkornel/Reproject-Image-To-3D
//saves pointcloud to .xyz file
void save(const cv::Mat& image3D, const std::string& fileName)
{
	CV_Assert(image3D.type() == CV_32FC3 && !image3D.empty());
	CV_Assert(!fileName.empty());

	std::ofstream outFile(fileName);

	if (!outFile.is_open())
	{
		std::cerr << "ERROR: Could not open " << fileName << std::endl;
		return;
	}

	for (int i = 0; i < image3D.rows; i++)
	{
		const cv::Vec3f* image3D_ptr = image3D.ptr<cv::Vec3f>(i);

		for (int j = 0; j < image3D.cols; j++)
		{
			outFile << image3D_ptr[j][0] << " " << image3D_ptr[j][1] << " " << image3D_ptr[j][2] << std::endl;
		}
	}

	outFile.close();
}


void computeCarteseanOccupancyMap(cv::Mat &src,cv::Mat &dst,const cv::Mat& Q,int gridSize, dimMinMax & gridDims,double heightPercentage)
{
	int grid_rows = (gridDims.max_z - gridDims.min_z )/gridSize  + 1;
	int grid_cols = (gridDims.max_x - gridDims.min_x )/gridSize + 1;
	cerr<<"Cartesean grid rows "<<grid_rows<<" Cartesean grid cols "<<grid_cols<<endl;

    dst = cv::Mat::zeros( grid_rows , grid_cols ,  CV_8U); //black means empty
    dst = Scalar(127);
    int k,m;
    float x,y,z;

    //for each pixel
    for (int i = 0; i <src.rows ; i++)
    {
        cv::Vec3f* out3D_ptr = src.ptr<cv::Vec3f>(i); // the point cloud 
        for (int j = 0; j < src.cols; j++)
        {
            cv::Vec3f& point = out3D_ptr[j];
            x = point[0] ;// get x
            y = point[1]  ;// get y
            z = point[2] ; //get z
            if(z ==0)
            	continue;
       //     cerr<<x<<" "<<y<<" "<<z<<endl; 
            //calulcate index in occupancy grid
            k = (x - gridDims.min_x)/gridSize;
            m = (z - gridDims.min_z)/gridSize;
            	//cerr<<"m(row) "<<m<<"k(col )"<<k<<"  "<<z<<"  " <<  gridDims.min_z <<endl;
         //   cerr<<k<<" "<<m<<endl;
            if(y <  THRESHOLD) //what is good threshold  
            	dst.at<char>(m,k) = 0; //obstcale
            else if(dst.at<char>(m,k) != 0)
            	dst.at<char>(m,k) = 255;

        }
    }
}
void computePolarOccupancyMap(cv::Mat &src,cv::Mat &dst,cv::Mat &output,const cv::Mat& Q,int gridSize, dimMinMax & gridDims,int imrows)
{
	int grid_rows = (gridDims.max_z - gridDims.min_z )/gridSize  + 1;
	int grid_cols = src.cols ;
	Mat test  = cv::Mat::zeros( src.rows , src.cols ,  CV_8UC3); //black means empty
	Mat temp = Mat::zeros(  imrows - src.rows, src.cols ,  CV_8UC3);
 
	cerr<<"Polar rows "<<grid_rows<<" Polar grid cols "<<grid_cols<<endl;

    dst = cv::Mat::zeros( grid_rows , grid_cols ,  CV_8U); //black means empty
    dst = Scalar(127);
    int k,m;
    float x,y,z;

    //for each pixel
    for (int i = 0; i <src.rows ; i++)
    {
        cv::Vec3f* out3D_ptr = src.ptr<cv::Vec3f>(i); // the point cloud 
        for (int j = 0; j < src.cols; j++)
        {
        	Vec3b &intensity = test.at<Vec3b>(i, j);
            cv::Vec3f& point = out3D_ptr[j];
            x = j;// get x
            y = point[1]  ;// get y
            z = point[2] ; //get z
            if(z ==0)
            	continue;
       //     cerr<<x<<" "<<y<<" "<<z<<endl; 
            //calulcate index in occupancy grid
            k = x;
            m = (z - gridDims.min_z)/gridSize;
            	//cerr<<"m(row) "<<m<<"k(col )"<<k<<"  "<<z<<"  " <<  gridDims.min_z <<endl;
         //   cerr<<k<<" "<<m<<endl;
             //y dimension in point cloud is inverted
            // also is the street is inclined 
            //900 -> 600 , 1800 -> 800
            if(y <=  THRESHOLD) //what is good threshold  ??
            {
            	dst.at<char>(m,k) = 0; //obstcale
            	intensity.val[2] = 255;
            }
            else if(dst.at<char>(m,k) != 0)
            {
            	dst.at<char>(m,k) = 255;
          		//intensity.val[1] = 255;  //no obstcale
	        }
        }
    } 
   if(temp.rows != 0)
   	vconcat(temp,test,output);
   else
	output = test;
    //imshow("test",test);
    //waitKey(0);

}
void computePolarOccupancyMap(cv::Mat &src,cv::Mat &dst,cv::Mat &output, vector< vector<int> >& Indecies,const cv::Mat& Q,int gridSize, dimMinMax & gridDims,int imrows)
{
    int grid_rows = (gridDims.max_z - gridDims.min_z )/gridSize  + 1;
    int grid_cols = src.cols ;
    Mat test  = cv::Mat::zeros( src.rows , src.cols ,  CV_8UC3); //black means empty
    Mat temp = Mat::zeros(  imrows - src.rows, src.cols ,  CV_8UC3);
    cerr<<"Polar rows "<<grid_rows<<" Polar grid cols "<<grid_cols<<endl;
    Indecies.resize(grid_cols*grid_rows);
    dst = cv::Mat::zeros( grid_rows , grid_cols ,  CV_8U); //black means empty
    dst = Scalar(127);
    int k,m;
    float x,y,z;

    //for each pixel
    for (int i = 0; i <src.rows ; i++)
    {
        cv::Vec3f* out3D_ptr = src.ptr<cv::Vec3f>(i); // the point cloud 
        for (int j = 0; j < src.cols; j++)
        {
            Vec3b &intensity = test.at<Vec3b>(i, j);
            cv::Vec3f& point = out3D_ptr[j];
            x = j;// get x
            y = point[1]  ;// get y
            z = point[2] ; //get z
            if(z ==0)
                continue;
       //     cerr<<x<<" "<<y<<" "<<z<<endl; 
            //calulcate index in occupancy grid
            k = x;
            m = (z - gridDims.min_z)/gridSize;
            Indecies[m*grid_cols + k].push_back(i);
            //cout<<Indecies[m*grid_cols + k].back()<<endl;
                //cerr<<"m(row) "<<m<<"k(col )"<<k<<"  "<<z<<"  " <<  gridDims.min_z <<endl;
         //   cerr<<k<<" "<<m<<endl;
             //y dimension in point cloud is inverted
            // also is the street is inclined 
            //900 -> 600 , 1800 -> 800
            if(y <=  THRESHOLD) //what is good threshold  ??
            {
                dst.at<char>(m,k) = 0; //obstcale
                intensity.val[2] = 255;
            }
            else if(dst.at<char>(m,k) != 0)
            {
                dst.at<char>(m,k) = 255;
                //intensity.val[1] = 255;  //no obstcale
            }
        }
    } 
   if(temp.rows != 0)
    vconcat(temp,test,output);
   else
    output = test;

    //imshow("test",test);
    //waitKey(0);
}
void segmentThresholdImg( cv::Mat &mask,double heightPercentage)
{
    int obstcale_flag= 0 ;
    for (int j = 0; j < mask.cols; j++) 
    {
		for (int i = mask.rows - 1 ; i >= mask.rows - heightPercentage*mask.rows  ; i--) //should be reversed 
    	{
        	Vec3b &intensity = mask.at<Vec3b>(i, j);
        	if((intensity.val[2] != 255) && (obstcale_flag == 0)) //no obstacle
        	{
        		intensity.val[1] = 255;
        	}
        	else if((intensity.val[2] != 255) && obstcale_flag ==1) //obstcale 
        	{
        		intensity.val[2] = 255;
        	}
        	else 
        	{
        		obstcale_flag = 1;
        	}
        }
        obstcale_flag = 0 ;
    }
}	

void projectPolarToImage(cv::Mat &grid,cv::Mat &dst, dimMinMax & gridDims,const cv::Mat& Q, int gridSize)
{
    int u,v,jpcl,imrows,imcols;
    float yc,zc,xc;
    float Q13 = Q.at<float>(1, 3);
    float Q23 = Q.at<float>(2, 3);
    imcols = dst.cols;
    imrows = dst.rows;
    //i and j are grid coordinates
    for(int i = 0 ; i < grid.rows;i++)
    { 
        //cerr<<"i = "<<i<<endl;
        for(int j=0; j < grid.cols;j++ )
        {
            u = j;
            if(grid.at<unsigned char>(i,j) == 0)//obstcale
            {
                yc = gridDims.min_y + THRESHOLD/2    ;
            }
            else if(grid.at<unsigned char>(i,j) == 255)
            {
                yc = gridDims.max_y ;
            }
            else continue;
            for(int k = 0 ; k < gridSize;k++)
            {
                zc = gridSize*i+gridDims.min_z + k;
                if(zc>gridDims.max_z) zc = gridDims.max_z;
                v = ((Q23*yc)/zc) - Q13 -1; 
                //cerr<<"u "<<u<<" v "<<v<<endl;
                if((v <= imrows ) && (v>= imrows - imrows*0.4))
                {
                    dst.at<unsigned char>(v,u) = grid.at<unsigned char>(i,j);
                    
                }
                  //else cerr<<"out "<<v<<endl;
    
            }
        }
    }
    /*it fill empty pixel in image, propably will not need this part after segementaion*/
    int flag_before = 0;
    for(int j=0; j < imcols;j++ )
    { 
        for(int i = imrows-1  ; i > imrows -0.4*imrows ;i--)   
        {
            if((dst.at<unsigned char>(i,j) != 0) && (dst.at<unsigned char>(i,j) != 255))
            {
                dst.at<unsigned char>(i,j) = flag_before;
            }
            else if(dst.at<unsigned char>(i,j) == 255) flag_before =255;
            else if(dst.at<unsigned char>(i,j) ==0 ) flag_before =0;

        }
    }
}
void projectPolarToImage(cv::Mat &grid,cv::Mat &dst, std::vector< std::vector<int> >& Indecies,double heightPercentage)
{
    //need to empty vector for next time 
    int u,v,offset,imrows,imcols;
    imcols = dst.cols;
    imrows = dst.rows;
    offset = (1-heightPercentage)*imrows;
    //i and j are grid coordinates
    for(int i = 0 ; i < grid.rows;i++)
    { 
        for(int j=0; j < grid.cols;j++ )
        {
            u = j;
            for(int c = 0 ; c<SIZE(Indecies[i*grid.cols +j]); c++)
                {
                    v = Indecies[i*grid.cols +j][c]  + offset;
                    dst.at<unsigned char>(v,u) = grid.at<unsigned char>(i,j);
                }
        }
    }
}
/*for(int i = 0 ; i < polarGrid2D.rows ; i ++)
{
    for(int j = 0 ; j < polarGrid2D.cols;j++)
    {
        for(int c = 0 ; c<SIZE(Indecies[i*polarGrid2D.cols +j]); c++)
        {
           cout<< Indecies[i*polarGrid2D.cols +j][c]<<endl;
        }
    }
}*/

void segementDP(cv::Mat &grid,cv::Mat &gridout)
{
    //search for first obstcale in depth direction
    //all the space before is considerd free space
    //calculate cost for each cell
    //Graph G(V,E) is generated 
    //V vertices, contains one vertx for each cell in the grid
    //E is set od edges which connect each vertex of one column, with each vertex of the nex column
    //Every edge has an assoicated values, which defines cost of segementaion
    //the cost contains 2 terms data and a smoothness
    //C(i,j,k,l) = Ed(i,j) + Es(i,j,k,l)
    //Ed(i,j) = 1/D(i,j) , D is the grid, if 0 -> obstcale -> cost -> inf
    //Es(i,j,k,l) = S(j,l) + T(i,j)
    //spatial part S(j,l) ,  penalizes jumps depth 
    // if S(j,l) > Thershold = CostParameter*distance 
    // if S(j,l) < Thershold = CostParameter*Thershold
    //temporal term T(i,j), depends on Ego motion (leave it now)
    //http://www.ias.informatik.tu-darmstadt.de/uploads/Theses/Zhou_MScThesis_2012.pdf
    //http://www.lelaps.de/papers/badino_wdv2007.pdf
    gridout = cv::Mat::zeros( grid.rows , grid.cols ,  CV_8U); //black means empty
    gridout = Scalar(127);
    double Ed,Es,S,Cs =0.00001,min_cost=10000.0;
    double Cost[grid.rows];
    int rowIndex[grid.cols];
    int D,D_THERSHOLD = 10,min_index = 0;
    //find start
    rowIndex[0] = 0; 
    for(int i = 0 ; i < grid.rows;i++)  
    {
        if(grid.at<unsigned char>(i,0) == 0)
        {
            rowIndex[0] = i;
            break;
        }
    }
    for(int j = 0 ; j <grid.cols -1 ; j++)
    {
        //for next column calculate D
        min_cost = 10000.0;
        for(int k = 0 ; k <grid.rows;k++)
        {
            if(grid.at<unsigned char>(k,j+1) == 0) //obstcale
            {
                Cost[k] = 10000;
            }
            else 
            {
                D = abs(rowIndex[j]- k); //to compensate for negative values
                Ed = 1.0/(double)(grid.at<unsigned char>(k,j+1));// Ed = [0.0039,0.0078]
                if(D <= D_THERSHOLD)
                {
                    S = Cs * D_THERSHOLD; //[0.0005]
                }
                else
                {
                    S = Cs * D; //[]
                }
            }
            Cost[k] = Ed + S;
            if(Cost[k] <= min_cost) 
            {
                min_cost = Cost[k];
                min_index = k;
            }
        }
        rowIndex[j+1] = min_index;
    }
    for(int i = 0 ; i < grid.rows;i++)
    {
        for(int j = 0; j < grid.cols ; j++)
        {
            if(rowIndex[j] <= i) gridout.at<unsigned char>(i,j) =0;
            else  gridout.at<unsigned char>(i,j) = 255;
        }
    }
}
void segmentThreshold(cv::Mat &grid,cv::Mat &gridout)
{
    int obstcale_flag= 0 ;
    gridout = cv::Mat::zeros( grid.rows , grid.cols ,  CV_8U); //black means empty
    for (int j = 0; j < grid.cols; j++) 
    {
        for (int i = 0  ; i < grid.rows  ; i++) //
        {;
            if((grid.at<unsigned char>(i,j) != 0) && (obstcale_flag == 0)) //no obstacle
            {
                gridout.at<unsigned char>(i,j) = 255;
            }
            else if((grid.at<unsigned char>(i,j) != 0) && obstcale_flag ==1) //obstcale 
            {
                gridout.at<unsigned char>(i,j) = 0;
            }   
            else if(grid.at<unsigned char>(i,j) == 0)
            {
                gridout.at<unsigned char>(i,j) = 0;
                obstcale_flag = 1;
            }
        }
        obstcale_flag = 0 ;
    }
}   