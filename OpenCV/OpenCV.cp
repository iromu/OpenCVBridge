/*
 *  OpenCV.cp
 *  OpenCV
 *
 *  Created by wantez on 07/06/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include <iostream>
#include "OpenCV.h"
#include "OpenCVPriv.h"

void OpenCV::HelloWorld(const char * s)
{
	 OpenCVPriv *theObj = new OpenCVPriv;
	 theObj->HelloWorldPriv(s);
	 delete theObj;
};

void OpenCVPriv::HelloWorldPriv(const char * s) 
{
	std::cout << s << std::endl;
};

