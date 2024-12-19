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
    int dw = (sizeDst.width - sizeEnlarge.width)/2;
    int dh = (sizeDst.height - sizeEnlarge.height)/2;

    cv::Mat M = cv::Mat::zeros(2,3,CV_32F);
    M.at<float>(0,0) = r;
    M.at<float>(1,1) = r;
    M.at<float>(0,2) = dw;
    M.at<float>(1,2) = dh;
    std::cout << M << std::endl;
    cv::warpAffine(src,src,M,(cv::Size(640,640)),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(114,114,114));
    src.copyTo(output) ;

}

void objectDetection::HWC2NormalCHW(cv::Mat input, std::vector<float> &data)
{
    std::vector<cv::Mat> inputChannel(3);
    cv::Mat img;
    cv::cvtColor(input, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0f/255.f);
    cv::split(img, inputChannel);
    std::vector<float>result;
    for(int i = 0; i < 3; i++)
    {
        std::vector<float> temp = std::vector<float>(inputChannel[i].reshape(1, 1));
        result.insert(result.end(),temp.begin(),temp.end());
    }
    data = result;
}

float objectDetection::areaBox(PredBox box)
{
    return box.width * box.height;
}

float objectDetection::iou(PredBox box1, PredBox box2)
{
    struct PredBox iouBox;
    float left = std::max(box1.cx-box1.width/2, box2.cx-box2.width/2);
    float top = std::max(box1.cy-box1.height/2,box2.cy-box2.height/2);
    float right = std::max(box1.cx+box1.width/2, box2.cx+box2.width/2);
    float down = std::max(box1.cy+box1.height/2,box2.cy+box2.height/2);

    float unionArea = std::max(right - left,float(0.0)) * \
                      std::max(down - top,float(0.0));
    float crossArea = areaBox(box1) + areaBox(box2) - unionArea;
    if(crossArea == 0.0 || unionArea == 0.0)
    {
        return 0.0;
    }
    return unionArea/crossArea;
}

void objectDetection::postprocess(cv::Mat &input)
{

    // Decode model output
    for(int i = 0; i < input.cols; i++)
    {
        cv::Mat res = input.col(i);

        float cx = res.at<float>(0);
        float cy = res.at<float>(1);
        float w = res.at<float>(2);
        float h = res.at<float>(3);
        cv::Mat cls = res(cv::Range(4,84),cv::Range::all());
        cv::Point minIdx, maxIdx;
        double minVal, maxVal;
        cv::minMaxLoc(cls, &minVal, &maxVal, &minIdx, &maxIdx);
        if(maxVal > m_conf_thres)
        {
            PredBox pbox;
            pbox.cx = cx;
            pbox.cy = cx;
            pbox.width = w;
            pbox.height = h;
            pbox.score = maxVal;
            pbox.label = maxIdx.y;
            m_PredBox_vector.push_back(pbox);
        }

        // break;
    }

    // Sort with score
    if(m_PredBox_vector.size()>0)
    {
        std::sort(m_PredBox_vector.begin(), m_PredBox_vector.end(), [](PredBox a, PredBox b)
        {
            return a.score > b.score;
        });
    }
    
}
