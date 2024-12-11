#include "../include/utils.hpp"

objectDetection::objectDetection(){};

objectDetection::~objectDetection(){};

void objectDetection::preprocess(cv::Mat input, cv::Mat &output)
{
    cv::Mat src = input.clone();
    cv::Size sizeDst(640,640);
    cv::Size sizeSrc = src.size();
    float ratioWidth = float(sizeDst.width)/float(sizeSrc.width);
    float ratioHeight = float(sizeDst.height)/float(sizeSrc.height);
    float r = ratioWidth < ratioHeight ? ratioWidth : ratioHeight;
    cv::Size sizeEnlarge(std::round(r*sizeSrc.width),std::round(r*sizeSrc.height));

    // Compute padding
    int dw = sizeDst.width - sizeEnlarge.width;
    int dh = sizeDst.height - sizeEnlarge.height;

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
    cv::resize(src,src,sizeEnlarge,cv::INTER_LINEAR);

    // Construct image
    cv::Scalar color(114, 114, 114); // BGR 顏色
    output.create(cv::Size(sizeEnlarge.width+2*dw,sizeEnlarge.height+2*dh), CV_8UC3);
    output.setTo(color);
    cv::Rect roi(dw,dh,src.cols,src.rows);
    src.copyTo(output(roi)) ;
}

void objectDetection::HWC2NormalCHW(cv::Mat input, std::vector<float> &data)
{
    std::vector<cv::Mat> inputChannel(3);
    cv::Mat img;
    cv::cvtColor(input, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3, 1.0f/255.f);
    cv::split(img, inputChannel);
    std::vector<float>result;
    for(int i = 0; i < 3; i++)
    {
        std::vector<float> temp = std::vector<float>(inputChannel[i].reshape(1, 1));
        result.insert(result.end(),temp.begin(),temp.end());
    }
    data = result;
}
