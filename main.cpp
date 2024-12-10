#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "src/utils.hpp"
int main()
{
    // Read image
    cv::Mat img = cv::imread("./data/horses.jpg");

    // Compute ratio
    cv::Size sizeDst(640,640);
    cv::Size sizeSrc = img.size();
    float ratioWidth = float(sizeDst.width)/float(sizeSrc.width);
    float ratioHeight = float(sizeDst.height)/float(sizeSrc.height);
    float r = ratioWidth < ratioHeight ? ratioWidth : ratioHeight;
    cv::Size sizeEnlarge(std::round(r*sizeSrc.width),std::round(r*sizeSrc.height));

    // Compute padding
    int stride = 32;
    int dw = sizeDst.width - sizeEnlarge.width;
    int dh = sizeDst.height - sizeEnlarge.height;

    bool autoFlag = false;
    bool scaleFillFlag = false;
    if(autoFlag)
    {
        dw = (sizeDst.width - sizeEnlarge.width)%stride;
        dh = (sizeDst.height - sizeEnlarge.height)%stride;
    }else if(scaleFillFlag)
    {
        dw = 0;
        dh = 0;
        sizeEnlarge = sizeDst;
        r = std::min(sizeEnlarge.width/sizeSrc.width, sizeEnlarge.height/sizeSrc.height);
    }
    dw/=2;
    dh/=2;
    // Resioz
    cv::resize(img,img,sizeEnlarge,cv::INTER_LINEAR);

    // Construct image
    cv::Scalar color(114, 114, 114); // BGR 顏色
    cv::Mat dst(cv::Size(sizeEnlarge.width+2*dw,sizeEnlarge.height+2*dh), CV_8UC3, color);
    cv::Rect roi(dw,dh,img.cols,img.rows);
    img.copyTo(dst(roi)) ;
    cv::imwrite("./data/test.jpg",dst);
    std::cout << r << std::endl;
}
