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

//typedef Scalar_<double> Scalar;
class OpenCV
{
public:
    //std::vector<cv::KeyPoint> keypoints1;
    //cv::Mat descriptors1;
    
    void init();
    void init(const char *,const char *);
    
    void saveBinaryKeyFile(const char *);
    void saveAsciiKeyFile(const char *);
    
    void feature_detect(int , int , int  ,  unsigned char * ,const char *);
    void feature_detect(const char *);
    void matcher(int , int , int  ,  unsigned char * );
};

#pragma GCC visibility pop
#endif
