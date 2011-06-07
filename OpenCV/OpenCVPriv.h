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

class OpenCVPriv
{
public:
    int getMatcherFilterType( const char * );
    
};

#pragma GCC visibility pop
