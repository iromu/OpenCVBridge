/*
 *  OpenCVPriv.h
 *  OpenCV
 *
 *  Created by wantez on 02/04/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

/* The classes below are not exported */

#pragma GCC visibility push(hidden)

#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

class OpenCVPriv
{
   
public:
   /*
    struct PointModel
    {
        vector<Point3f> points;
        vector<vector<int> > didx;
        Mat descriptors;
        string name;
    };
    */
    int getMatcherFilterType( const string& );
    void simpleMatching( Ptr<DescriptorMatcher>& , const Mat& , const Mat& ,vector<DMatch>&  );
    void crossCheckMatching( Ptr<DescriptorMatcher>& , const Mat& , const Mat& , vector<DMatch>& , int );
    
    /*
    static void build3dmodel( const Ptr<FeatureDetector>& ,
                             const Ptr<DescriptorExtractor>& ,
                             const vector<Point3f>& ,
                             const vector<string>& ,
                             const vector<Rect>& ,
                             const vector<Vec6f>& ,
                             const Mat& ,
                             PointModel&  );
    
    static Mat getFundamentalMat( const Mat& , const Mat& ,
                                             const Mat& , const Mat& ,
                                             const Mat&  );
    
    void findConstrainedCorrespondences(const Mat& ,
                                                    const vector<KeyPoint>& ,
                                                    const vector<KeyPoint>& ,
                                                    const Mat& ,
                                                    const Mat& ,
                                                    vector<Vec2i>& ,
                                                    double , double );
    
    void unpackPose(const Vec6f& , Mat& , Mat& );
    */
};

#pragma GCC visibility pop
