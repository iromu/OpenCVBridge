/*
 *  OpenCV.cp
 *  OpenCV
 *
 *  Created by wantez on 02/04/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>


#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iomanip>

#include "OpenCV.h"
#include "OpenCVPriv.h"

using namespace cv;
using namespace std;

#define DRAW_RICH_KEYPOINTS_MODE    0
#define DRAW_OUTLIERS_MODE           0

const string winName = "correspondences";

enum { NONE_FILTER = 0, CROSS_CHECK_FILTER = 1 };

bool isWarpPerspective = false;
double ransacReprojThreshold = -1; 

Ptr<FeatureDetector> detector;
Ptr<DescriptorExtractor> descriptorExtractor;
Ptr<DescriptorMatcher> descriptorMatcher;


Mat img1;

int mactherFilterType;
string detectorString;
string extractorString;
string algStringCode;

std::vector<KeyPoint> keypoints1;
Mat descriptors1;


void OpenCV::init()
{    
    init("SIFT","SIFT");
};

void OpenCV::init(const char *  d,const char *  e)
{
    int max_threads = cvGetNumThreads();
    max_threads = cv::getNumThreads();
    int threadnum = cv::getThreadNum();
    detectorString = d;
    extractorString = e;
    
    std::stringstream algCode;
    algCode<<d<<"-"<<e;
    algStringCode = (d == e)?d:algCode.str().c_str();
    
    isWarpPerspective=false;
    //    ransacReprojThreshold=3;
    
    cout << "< Creating detector";
    detector = FeatureDetector::create( detectorString );
    
    cout << "< Creating detector, descriptor extractor";
    descriptorExtractor = DescriptorExtractor::create( extractorString );
    
    cout << "< Creating detector, descriptor extractor and descriptor matcher ... ";
    descriptorMatcher = DescriptorMatcher::create( "FlannBased" );
    
    OpenCVPriv *theObj = new OpenCVPriv;
    mactherFilterType = theObj->getMatcherFilterType( "CrossCheckFilter" );
    cout << ">" << endl;
    if( detector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty()  )
    {
        cout << "Can not create detector or descriptor extractor or descriptor matcher of given types" << endl;
        //return -1;
    }
    
};

void OpenCV::matcher(int h, int w, int  samplesPerPixel,  unsigned char * bitmapData)
{
    if(!img1.empty())   {
        Mat img2 =  Mat(h,w,CV_MAKETYPE(CV_8U,  samplesPerPixel),  bitmapData);
        RNG rng = theRNG();
        Mat H12;
        if( isWarpPerspective ){
            //warpPerspectiveRand(img1, img2, H12, rng );
            
            H12.create(3, 3, CV_32FC1);
            H12.at<float>(0,0) = rng.uniform( 0.8f, 1.2f);
            H12.at<float>(0,1) = rng.uniform(-0.1f, 0.1f);
            H12.at<float>(0,2) = rng.uniform(-0.1f, 0.1f)*img1.cols;
            H12.at<float>(1,0) = rng.uniform(-0.1f, 0.1f);
            H12.at<float>(1,1) = rng.uniform( 0.8f, 1.2f);
            H12.at<float>(1,2) = rng.uniform(-0.1f, 0.1f)*img1.rows;
            H12.at<float>(2,0) = rng.uniform( -1e-4f, 1e-4f);
            H12.at<float>(2,1) = rng.uniform( -1e-4f, 1e-4f);
            H12.at<float>(2,2) = rng.uniform( 0.8f, 1.2f);
            
            warpPerspective( img1, img2, H12, img1.size() );
            
        }
        bool eval = !isWarpPerspective ? false : (ransacReprojThreshold == 0 ? false : true);
        
        cout << endl << "< Extracting keypoints from second image... ";
        vector<KeyPoint> keypoints2;
        detector->detect( img2, keypoints2 );
        cout << keypoints2.size() << " points >" << endl;
        
        if( !H12.empty() && eval )  {
            cout << "< Evaluate feature detector..." << endl;
            float repeatability;
            int correspCount;
            evaluateFeatureDetector( img1, img2, H12, &keypoints1, &keypoints2, repeatability, correspCount );
            cout << "repeatability = " << repeatability << endl;
            cout << "correspCount = " << correspCount << endl;
            cout << ">" << endl;
        }
        
        cout << "< Computing descriptors for keypoints from second image... ";
        Mat descriptors2;
        descriptorExtractor->compute( img2, keypoints2, descriptors2 );
        cout << ">" << endl;
        
        cout << "< Matching descriptors... ";
        vector<DMatch> filteredMatches;
        
        OpenCVPriv *theObj = new OpenCVPriv;
        delete theObj;
        
        switch( mactherFilterType )
        {
            case CROSS_CHECK_FILTER :
            {
                int knn = 1;
                filteredMatches.clear();
                vector<vector<DMatch> > matches12, matches21;
                descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
                descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
                for( size_t m = 0; m < matches12.size(); m++ )
                {
                    bool findCrossCheck = false;
                    for( size_t fk = 0; fk < matches12[m].size(); fk++ )
                    {
                        DMatch forward = matches12[m][fk];
                        
                        for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
                        {
                            DMatch backward = matches21[forward.trainIdx][bk];
                            if( backward.trainIdx == forward.queryIdx )
                            {
                                filteredMatches.push_back(forward);
                                findCrossCheck = true;
                                break;
                            }
                        }
                        if( findCrossCheck ) break;
                    }
                }
            }
                break;
            default :{
                vector<DMatch> matches;
                descriptorMatcher->match( descriptors1, descriptors2, filteredMatches );
            }
        }
        cout << ">" << endl;
        
        if( !H12.empty()  &&eval)
        {
            cout << "< Evaluate descriptor match..." << endl;
            vector<Point2f> curve;
            Ptr<GenericDescriptorMatcher> gdm = new VectorDescriptorMatcher( descriptorExtractor, descriptorMatcher );
            evaluateGenericDescriptorMatcher( img1, img2, H12, keypoints1, keypoints2, 0, 0, curve, gdm );
            for( float l_p = 0; l_p < 1 - FLT_EPSILON; l_p+=0.1f )
                cout << "1-precision = " << l_p << "; recall = " << getRecall( curve, l_p ) << endl;
            cout << ">" << endl;
        }
        
        vector<int> queryIdxs( filteredMatches.size() ), trainIdxs( filteredMatches.size() );
        for( size_t i = 0; i < filteredMatches.size(); i++ )
        {
            queryIdxs[i] = filteredMatches[i].queryIdx;
            trainIdxs[i] = filteredMatches[i].trainIdx;
        }
        
        if( !isWarpPerspective && ransacReprojThreshold >= 0 )
        {
            cout << "< Computing homography (RANSAC)..." << endl;
            vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, queryIdxs);
            vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, trainIdxs);
            H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold );
            cout << ">" << endl;
        }
        
        Mat drawImg;
        if( !H12.empty() ) // filter outliers
        {
            vector<char> matchesMask( filteredMatches.size(), 0 );
            vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, queryIdxs);
            vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, trainIdxs);
            Mat points1t; perspectiveTransform(Mat(points1), points1t, H12);
            for( size_t i1 = 0; i1 < points1.size(); i1++ )
            {
                if( norm(points2[i1] - points1t.at<Point2f>((int)i1,0)) < 4 ) // inlier
                    matchesMask[i1] = 1;
            }
            
            // draw inliers
            drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask
#if DRAW_RICH_KEYPOINTS_MODE
                        , DrawMatchesFlags::DRAW_RICH_KEYPOINTS
#endif
                        );
            
#if DRAW_OUTLIERS_MODE
            // draw outliers
            for( size_t i1 = 0; i1 < matchesMask.size(); i1++ )
                matchesMask[i1] = !matchesMask[i1];
            drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg, CV_RGB(0, 0, 255), CV_RGB(255, 0, 0), matchesMask,
                        DrawMatchesFlags::DRAW_OVER_OUTIMG | DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
#endif
        }
        else
            drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg );
        
        //drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg );
        
        
        //namedWindow(winName, 1);
        imshow( winName, drawImg );
        
    }
}

void OpenCV::feature_detect(int h, int w, int  samplesPerPixel,  unsigned char * bitmapData, const char *  filename)
{
    
    cout << endl << "< Extracting keypoints from image... " ;
    Mat img;
    //filename=0;
    if (filename) {
        img=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
        // Mat color_dst;
        //cvtColor( img, color_dst, CV_GRAY2BGR );
        namedWindow(winName, CV_WINDOW_NORMAL);
        setWindowProperty(winName,CV_WND_PROP_ASPECTRATIO,CV_WINDOW_KEEPRATIO);
        imshow( winName, img );
    }
    else{
        //BUG
        img =  Mat(h,w,CV_MAKETYPE(CV_8U,  samplesPerPixel),  bitmapData);
        
        namedWindow(winName, CV_WINDOW_NORMAL);
        setWindowProperty(winName,CV_WND_PROP_ASPECTRATIO,CV_WINDOW_KEEPRATIO);
        imshow( winName, img );
    }
    
    img1=img;
    vector<KeyPoint> keypoints;
    keypoints1 = keypoints;
    
    detector->detect( img, keypoints1 );
    cout << keypoints1.size() << " points >" << endl;
    
    cout << "< Computing descriptors for keypoints from image... ";
    
    Mat descriptors;
    descriptors1=descriptors;
    
    descriptorExtractor->compute( img, keypoints1, descriptors1 );
    cout << keypoints1.size() << " total points + ";
    cout << descriptors1.total() << " descriptors >" << endl;
    if (filename)
    {
        std::stringstream filepath,filepathKeypoints,filepathDescriptors;
        filepath << filename << "." << algStringCode << ".xml.gz";
        //filepathKeypoints << filename << "." << algStringCode << ".keypoints.xml.gz";
        //filepathDescriptors << filename << "." << algStringCode << ".descriptors.xml.gz";
        
        
        //saveAsciiKeyFile(filename);
        cv::FileStorage kfs(filepath.str().c_str(), cv::FileStorage::WRITE);
        if( kfs.isOpened())
        {
            //descriptors1.write(dfs);
            cv::write(kfs, "algorithm", algStringCode);
            cv::write(kfs, "keypoints", keypoints1);
            //kfs.release();
            
            //cv::FileStorage dfs(filepathDescriptors.str().c_str(), cv::FileStorage::WRITE);
            //descriptors1.write(dfs);
            //if( dfs.isOpened())
            cv::write(kfs, "descriptors", descriptors1);
        }
        //dfs.release();
        kfs.release();
        //dfs.write(dfs, keypoints1);
        // saveBinaryKeyFile(filename);
    }
    //namedWindow(winName, 1);
    //imshow( winName, descriptors1 );
};


void OpenCV::feature_detect( const char *  filename)
{
    cout << endl << "< Extracting keypoints from image " << filename ;
    
    Mat img;
    img=imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    img1=img;
    //namedWindow(winName);
    //setWindowProperty(winName,CV_WND_PROP_ASPECTRATIO,CV_WINDOW_KEEPRATIO);
    //imshow( winName, img );
    
    
    vector<KeyPoint> keypointsd;
    keypoints1 = keypointsd;
    
    detector->detect( img1, keypoints1 );
    
    cout << keypoints1.size() << " points >" << endl;
    
    cout << "< Computing descriptors for keypoints from image... ";
    
    Mat descriptorsd;
    descriptors1=descriptorsd;
    
    descriptorExtractor->compute( img1, keypoints1, descriptors1 );
    cout << keypoints1.size() << " total points + ";
    cout << descriptors1.total() << " descriptors >" << endl;
    
    std::stringstream output;
    output << filename << "." << algStringCode << ".xml.gz";
    cv::FileStorage fs(output.str().c_str(), FileStorage::WRITE);
    if( fs.isOpened())
    {
        cv::write(fs, "algorithm", algStringCode);
        cv::write(fs, "keypoints", keypoints1);
        cv::write(fs, "descriptors", descriptors1);
    }
    fs.release();
};




void OpenCV::saveAsciiKeyFile(const char *  filename)
{	
	std::stringstream filepath;
	filepath << filename << "." << algStringCode << ".key.txt";
    
	std::ofstream output(filepath.str().c_str());
	if (output.is_open())
    {
		output.flags(std::ios::fixed);
        
		//const FeatureInfo& info = mFeatureInfos[fileIndex];
		//unsigned int nbFeature = (unsigned int) info.points.size();
		unsigned int nbFeature = (unsigned int) keypoints1.size();
		
        //const float* pd = &info.descriptors[0];
		
		output << nbFeature << " " << descriptors1.row(0).total() <<std::endl;
        
		for (unsigned int i=0; i<nbFeature; ++i)
        {
            //in y, x, scale, orientation order
			output << std::setprecision(2) << keypoints1[i].pt.y << " " << std::setprecision(2) << keypoints1[i].pt.x << " " << std::setprecision(3) << keypoints1[i].size << " " << std::setprecision(3) <<  keypoints1[i].angle << std::endl;
            
            // BUG
            uchar* descriptor = descriptors1.row(i).data;
			for (int k=0; k<descriptors1.row(i).total(); ++k, ++descriptor)
            {
				//output << ((unsigned int)floor(0.5+512.0f*(*descriptor)))<< " ";
                output << ((unsigned int)floor(0.5+(*descriptor)))<< " ";
                //  cout << "< Saving descriptor " << (*descriptor)  << endl;
				if ((k+1)%20 == 0) 
					output << std::endl;
            }
			output << std::endl;
        }
    }
	output.close();
};

void OpenCV::saveBinaryKeyFile(const char *  filename)
{
	std::stringstream filepath,filepathKeypoints,filepathDescriptors;
	filepath << filename << "." << algStringCode << ".key.bin";
    filepathKeypoints << filename << "." << algStringCode << ".keypoints";
    filepathDescriptors << filename << "." << algStringCode << ".descriptors";
    
    cout << "< Saving file " << filepath.str().c_str()  << endl;
    
    cout << keypoints1.size() << " total points + ";
    cout << descriptors1.total() << " descriptors >" << endl;
    
    std::ofstream kofs;
    kofs.open(filepathKeypoints.str().c_str(), std::ios::out | ios::binary);
    if (kofs.is_open())
        kofs.write((char *)&keypoints1, sizeof(KeyPoint)*keypoints1.size());
    kofs.close();
    
    std::ofstream dofs(filepathDescriptors.str().c_str(), std::ios::out | ios::binary);
    if (dofs.is_open())
        dofs.write((char *)&descriptors1, descriptors1.total() *descriptors1.elemSize());
    dofs.close();
    
    /*
     std::ofstream output;
     output.open(filepath.str().c_str(), std::ios::out | std::ios::binary);
     
     
     
     if (output.is_open())
     {
     
     int nbFeature = (int)keypoints1.size();//(int)mFeatureInfos[fileIndex].points.size();
     output.write((char*)&nbFeature, sizeof(nbFeature));
     
     // FeatureInfo featureInfo = mFeatureInfos[fileIndex];
     for (int i=0; i<nbFeature; ++i)
     {			
     float x           = keypoints1[i].pt.x;
     float y           = keypoints1[i].pt.y;
     float scale       = keypoints1[i].size;
     float orientation = keypoints1[i].angle;
     
     uchar* descriptor = descriptors1.row(i).data;
     
     
     
     output.write((char*)&x, sizeof(x));
     output.write((char*)&y, sizeof(y));
     output.write((char*)&scale, sizeof(scale));
     output.write((char*)&orientation, sizeof(orientation));
     // output.write((char*)descriptor, sizeof(float)*descriptors1.row(i).total());	
     output.write((char*)descriptor, sizeof(descriptor));	
     }
     }
     
     output.close();
     */
    
};


int OpenCVPriv::getMatcherFilterType( const char * str)
{
    if( str == "NoneFilter" )
        return NONE_FILTER;
    if( str == "CrossCheckFilter" )
        return CROSS_CHECK_FILTER;
    CV_Error(CV_StsBadArg, "Invalid filter name");
    return -1;
};

