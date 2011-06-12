/*
 *  OpenCV.h
 *  OpenCV
 *
 *  Created by wantez on 02/04/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef OpenCV_
#define OpenCV_



/* The classes below are exported */
#pragma GCC visibility push(default)
//#include <opencv2/core/core.hpp>
//typedef Scalar_<double> Scalar;
class OpenCV
{
public:
    //std::vector<cv::KeyPoint> keypoints1;
    //cv::Mat descriptors1;
    
    void init();
    
    void init(const char *,const char *);
    
    void initFeatureDetector(const char *);
    void initDescriptorExtractor(const char *);
    void initDescriptorMatcher(const char *,const char *);
    
    
    void matchFeatures(const char *,const char *);
    
    void saveBinaryKeyFile(const char *);
    void saveAsciiKeyFile(const char *);
    
    void feature_detect(const char *);
    void buildPointModel();
    
};

#pragma GCC visibility pop
#endif
