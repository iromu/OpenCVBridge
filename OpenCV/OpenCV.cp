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
#include <execinfo.h>
#include <stdio.h>

#include "OpenCV.h"
#include "OpenCVPriv.h"

#include "BuildModel.cpp"

using namespace cv;
using namespace std;

#define DRAW 1
#define DRAW_RICH_KEYPOINTS_MODE    0
#define DRAW_OUTLIERS_MODE           0

//#define DEBUG

const string winName = "verbose";

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

OpenCV::~OpenCV()
{
   drawProgress=false;
}

void OpenCV::init()
{    
    //drawProgress=false;
    //init("SIFT","SIFT");
};

void OpenCV::initDescriptorExtractor(const char *  d)
{    
    extractorString = d;
    descriptorExtractor = DescriptorExtractor::create( extractorString );
};

void OpenCV::initFeatureDetector(const char *  d)
{    
    detectorString = d;
    detector = FeatureDetector::create( detectorString );
};

void OpenCV::initDescriptorMatcher(const char *  dd,const char *  ff)
{    
    descriptorMatcher = DescriptorMatcher::create( dd );
    OpenCVPriv *theObj = new OpenCVPriv;
    mactherFilterType = theObj->getMatcherFilterType( ff );
    delete theObj;
};




unsigned long OpenCV::matchFeatures(const char *  ii,const char *  jj)

{
    std::stringstream output;
    output << ii<<".matches.xml";
    return matchFeatures(ii, jj,output.str().c_str());
}
unsigned long OpenCV::matchFeatures(const char *  ii,const char *  jj,const char *  dest)
{
    
    //try{
    
    int ransacReprojThreshold=8;//3 8, 1-10
    
    Mat iDescriptors, jDescriptors;
    string filename1,filename2;
    std::vector<KeyPoint> iKeypoints,jKeypoints;
    cv::FileStorage fs(ii, FileStorage::READ);
    if( fs.isOpened())
    {
        //fs["keypoints"] >> iKeypoints;
        
        fs["filename"] >> filename1;
        cv::read( fs["keypoints"], iKeypoints);
        fs["descriptors"] >> iDescriptors;
        
        cout << iDescriptors.total() << " iDescriptors >" ;
        
    }
    //fs.release();
    fs.open(jj, FileStorage::READ);
    if( fs.isOpened())
    {
        //fs["keypoints"] >> jKeypoints;
        fs["filename"] >> filename2;
        fs["descriptors"] >> jDescriptors;
        //cv::read("keypoints",  jKeypoints);
        cv::read( fs["keypoints"], jKeypoints);
        cout << jDescriptors.total() << " jDescriptors >" ;
        
    }
    //fs.release();
    
    OpenCVPriv *theObj = new OpenCVPriv;
    
    
    vector<DMatch> filteredMatches;
    switch( mactherFilterType )
    {
        case CROSS_CHECK_FILTER :
            theObj->crossCheckMatching( descriptorMatcher, iDescriptors, jDescriptors, filteredMatches, 1 );
            break;
        default :
            theObj->simpleMatching( descriptorMatcher, iDescriptors, jDescriptors, filteredMatches );
    }
    
    cout << filteredMatches.size() << " filteredMatches >" << endl;
    
    Mat H12;
    vector<int> queryIdxs( filteredMatches.size() ), trainIdxs( filteredMatches.size() );
    for( size_t i = 0; i < filteredMatches.size(); i++ )
    {
        queryIdxs[i] = filteredMatches[i].queryIdx;
        trainIdxs[i] = filteredMatches[i].trainIdx;
    }
    
    if(  ransacReprojThreshold >= 0 )
    {
        
        cout << "< Computing homography (RANSAC)..." << endl;
        vector<Point2f> points1; KeyPoint::convert(iKeypoints, points1, queryIdxs);
        vector<Point2f> points2; KeyPoint::convert(jKeypoints, points2, trainIdxs);
        H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold );
        cout << ">" << endl;
    }
    
    //std::stringstream output;
    //output << ii<<".matches.xml";
    fs.open(dest, FileStorage::WRITE);
    if( fs.isOpened())
    {
        
        // fs<<"filteredMatches"<<filteredMatches;
        //fs<<"filteredMatches"<<(DMatch)filteredMatches[0];
        // cv::write(fs, "filteredMatches", (DMatch)filteredMatches[0]);
        fs << "FEATURES1" << ii << "FEATURES2" << jj;
        fs << "matches" << "[";
        for (int i=0; i<filteredMatches.size(); ++i) {
            
            DMatch o = filteredMatches[i];
            fs << "{" << "queryIdx" << o.queryIdx << "trainIdx" << o.trainIdx << "imgIdx"
            << o.imgIdx << "distance"
            << o.distance << "}";
            
            //fs<<o.queryIdx<<o.trainIdx<<o.imgIdx<<o.distance;
        }
        fs << "]";
        
        
    }
    
    fs.release();
    
    delete theObj;
    
    //#if DRAW
    if(drawProgress){
        Mat drawImg;
        Mat img1=imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
        Mat img2=imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);
        
        
        
        if( !H12.empty() ) // filter outliers
        {
            vector<char> matchesMask( filteredMatches.size(), 0 );
            vector<Point2f> points1; KeyPoint::convert(iKeypoints, points1, queryIdxs);
            vector<Point2f> points2; KeyPoint::convert(jKeypoints, points2, trainIdxs);
            Mat points1t; perspectiveTransform(Mat(points1), points1t, H12);
            for( size_t i1 = 0; i1 < points1.size(); i1++ )
            {
                if( norm(points2[i1] - points1t.at<Point2f>((int)i1,0)) < 4 ) // inlier
                    matchesMask[i1] = 1;
            }
            // draw inliers
            drawMatches( img1, iKeypoints, img2, jKeypoints, filteredMatches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask
#if DRAW_RICH_KEYPOINTS_MODE
                        , DrawMatchesFlags::DRAW_RICH_KEYPOINTS
#endif
                        );
            
#if DRAW_OUTLIERS_MODE
            // draw outliers
            for( size_t i1 = 0; i1 < matchesMask.size(); i1++ )
                matchesMask[i1] = !matchesMask[i1];
            drawMatches( img1, iKeypoints, img2, jKeypoints, filteredMatches, drawImg, CV_RGB(0, 0, 255), CV_RGB(255, 0, 0), matchesMask,
                        DrawMatchesFlags::DRAW_OVER_OUTIMG | DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
#endif
        }
        else
            drawMatches( img1, iKeypoints, img2, jKeypoints, filteredMatches, drawImg );
        
        
        //    const int scale = 2;
        //cv::resize(const cv::Mat &src, <#cv::Mat &dst#>, <#Size dsize#>)
        //CvMat* cvmat = drawImg;
        //IplImage *  gray_image = cvmat; 
        //IplImage *  small_image   = cvCreateImage(cvSize (gray_image->width / scale, gray_image->height / scale), IPL_DEPTH_8U, 1);
        Mat small_image; 
        
        cv::resize(drawImg, small_image, img1.size());
        
        imshow( winName, small_image );
        //#endif 
    }
    
    
    
    return filteredMatches.size();
    /*  
     }
     
     catch (Exception theException) 
     {
     cout << "OPENCV_MATCH_EXCEPTION: ";
     theException.formatMessage();
     }
     catch(...) {
     cout << "OPENCV_MATCH_UNKNOWN_EXCEPTION" << endl;
     void* callstack[128];
     int i, frames = backtrace(callstack, 128);
     char** strs = backtrace_symbols(callstack, frames);
     for (i = 0; i < frames; ++i) {
     printf("%s\n", strs[i]);
     }
     free(strs);
     }*/
    return 0;
    
};

void OpenCV::init(const char *  d,const char *  e)
{
    detectorString = d;
    extractorString = e;
    
    std::stringstream algCode;
    algCode<<d<<"-"<<e;
    algStringCode = (d == e)?d:algCode.str().c_str();
    
    cout << "< Creating detector";
    detector = FeatureDetector::create( detectorString );
    
    cout << "< Creating detector, descriptor extractor";
    descriptorExtractor = DescriptorExtractor::create( extractorString );
    
    cout << ">" << endl;
    if( detector.empty() || descriptorExtractor.empty() )
    {
        cout << "Can not create detector or descriptor extractor of given types" << endl;
    }
    
};

unsigned long OpenCV::feature_check(const char * filename)
{
    std::vector<KeyPoint> keypoints;
    cv::FileStorage fs(filename, FileStorage::READ);
    if( fs.isOpened())
    {
        cv::read( fs["keypoints"], keypoints);
        return keypoints.size();
    }
    return 0;
    fs.release();
};

unsigned long OpenCV::match_check(const char * filename)
{
    vector<DMatch> filteredMatches;
    cv::FileStorage fs(filename, FileStorage::READ);
    unsigned long size=0;
    if( fs.isOpened())
    {
        FileNode n = fs["matches"];
        if (n.type() != FileNode::SEQ)
        {
            cerr << "images is not a sequence! FAIL" << endl;
            return 1;
        }
        
        cout << "reading matches\n";
        FileNodeIterator it = n.begin(), it_end = n.end();
        for (; it != it_end; ++it)
        {
            //            cv::read(fs, <#double &value#>, <#double default_value#>)
            
            /*A = (int)*it ["A"];
             X = (double)*it ["X"];
             id = (string)*it ["id"];
             */
            size++;
        }
        
        
        return size;
    }
    return 0;
    fs.release();
};
unsigned long OpenCV::feature_detect( const char *  filename)
{
    std::stringstream algCode;
    algCode<<detectorString<<"-"<<extractorString;
    algStringCode = (detectorString == extractorString)?detectorString:algCode.str().c_str();
    
    cout << endl << "< Extracting keypoints from image " << filename ;
    try {
        
        
        Mat img;
        img=imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
        //img1=img;
        //namedWindow(winName);
        //setWindowProperty(winName,CV_WND_PROP_ASPECTRATIO,CV_WINDOW_KEEPRATIO);
        //imshow( winName, img );
        
        
        vector<KeyPoint> keypoints;
        //keypoints1 = keypoints;
        
        detector->detect( img, keypoints );
        
        cout << keypoints.size() << " points >" << endl;
        
        cout << "< Computing descriptors for keypoints from image... ";
        
        //Mat descriptorsd;
        Mat descriptors(1, 
                        (int)(keypoints.size() * sizeof(KeyPoint)), CV_8U, 
                        &keypoints[0]);
        //descriptors1=descriptorsd;
        
        descriptorExtractor->compute( img, keypoints, descriptors );
        cout << keypoints.size() << " total points + ";
        cout << descriptors.total() << " descriptors >" << endl;
        
        std::stringstream output;
        output << filename << "." << algStringCode << ".xml";
        cv::FileStorage fs(output.str().c_str(), FileStorage::WRITE);
        if( fs.isOpened())
        {
            fs<<"filename"<<filename;
            fs<<"algorithm"<<algStringCode;
            fs<<"descriptors"<<descriptors;
            //fs<<"keypoints"<<keypoints;
            cv::write(fs, "keypoints", keypoints);
            /*
             fs << "keypoints" << "[";
             for(int i=0;i<keypoints.size();++i)
             {
             // fs << (KeyPoint)keypoints[i];
             //cv::write(fs, "keypoint", (KeyPoint)keypoints[i]);
             }
             
             //cv::write(fs, "keypoints", keypoints);
             
             fs << "]";
             */
            /* 
             cv::write(fs, "algorithm", algStringCode);
             cv::write(fs, "keypoints", keypoints);
             cv::write(fs, "descriptors", descriptors);
             */ 
        }
        
        fs.release();
        
        //cv::FileStorage->flush(output);
        //cvFlushSeqWriter(<#CvSeqWriter *writer#>)
        //#ifdef DEBUG
        //#if DRAW
        
        if(drawProgress){
            Mat outImg;
            drawKeypoints( img, keypoints, outImg);
            namedWindow(winName);
            imshow( winName, outImg );
        }
        //#endif
        
        
        return keypoints.size();
    }
    catch(...) {
        cout << "OPENCV_FEATURE_UNKNOWN_EXCEPTION" << endl;
    }
    return 0;
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
void OpenCV::buildPointModel(vector<string> imageList)
{
    // vector<string> imageList;
    vector<Rect> roiList;
    vector<Vec6f> poseList;
    
    vector<Point3f> modelBox;
    PointModel model;
    const char* modelName = "test";
    model.name = modelName;
    Mat cameraMatrix, distCoeffs;
    string intrinsicsFilename="/Users/Shared/VisualProcessing/data/camera.yml"  ;
    Size calibratedImageSize;
    readCameraMatrix(intrinsicsFilename, cameraMatrix, distCoeffs, calibratedImageSize);
    
    
    // OpenCVPriv *theObj = new OpenCVPriv;
    build3dmodel( detector, descriptorExtractor, modelBox,
                 imageList, roiList, poseList, cameraMatrix, model );
    string outputModelName = format("%s_model.yml.gz", modelName);
    
    //delete theObj;
};


void OpenCVPriv::simpleMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                                const Mat& descriptors1, const Mat& descriptors2,
                                vector<DMatch>& matches12 )
{
    vector<DMatch> matches;
    descriptorMatcher->match( descriptors1, descriptors2, matches12 );
};

void OpenCVPriv::crossCheckMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                                    const Mat& descriptors1, const Mat& descriptors2,
                                    vector<DMatch>& filteredMatches12, int knn=1 )
{
    filteredMatches12.clear();
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
                    filteredMatches12.push_back(forward);
                    findCrossCheck = true;
                    break;
                }
            }
            if( findCrossCheck ) break;
        }
    }
};

int OpenCVPriv::getMatcherFilterType( const string& str)
{
    //std::string s = str;
    if( str == "NoneFilter" )
        return NONE_FILTER;
    if( str == "CrossCheckFilter" )
        return CROSS_CHECK_FILTER;
    
    return NONE_FILTER;
    //CV_Error(CV_StsBadArg, "Invalid filter name");
    //return -1;
};

