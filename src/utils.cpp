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

float objectDetection::areaBox(PredBox &box)
{
    return box.width * box.height;
}

float objectDetection::iou(PredBox box1, PredBox box2)
{

    float left = std::max(box1.cx-box1.width/2, box2.cx-box2.width/2);
    float top = std::max(box1.cy-box1.height/2,box2.cy-box2.height/2);
    float right = std::min(box1.cx+box1.width/2, box2.cx+box2.width/2);
    float down = std::min(box1.cy+box1.height/2,box2.cy+box2.height/2);

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
            pbox.cy = cy;
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
    // NMS
    NMS(m_PredBox_vector);
}

void objectDetection::NMS(std::vector<PredBox> &pred)
{
    std::vector<PredBox> result;
    std::vector<bool> table(pred.size(),false);
    for(int i = 0; i < pred.size(); i++)
    {
        if(table[i])continue;
        PredBox temp = pred[i];
        result.push_back(temp);
        for(int j = i+1; j < pred.size(); j++)
        {
            if(table[j])continue;
            if(iou(temp,pred[j]) > m_iou_thres)
            {
                table[j] = true;
            }
        }
    }
    pred = result;
    for(int i = 0; i < pred.size(); i++)
    {
        std::cout << pred[i].score << std::endl;
    }
}

void objectDetection::draw(cv::Mat &input, cv::Mat &output, std::vector<PredBox> &pred)
{
    output = input.clone();
    float ration_w = 640/float(input.cols);
    float ration_h = 640/float(input.rows);
    for(int i = pred.size()-1; i >= 0 ; i--)
    {
        int left = int(pred[i].cx-pred[i].width/2);
        int top  = int(pred[i].cy-pred[i].height/2);
        int width = pred[i].width;
        int height = pred[i].height;
        if(ration_h > ration_w)
        {
            left = left /ration_w;
            top  = (top-(640-ration_w*input.rows)/2)/ration_w;
            width = width / ration_w;
            height = height / ration_w;
        }else
        {
            left = left /ration_h;
            top  = (top-(640-ration_h*input.cols)/2)/ration_h;
            width = width / ration_h;
            height = height / ration_h;
        }
        // Rectangle
        cv::rectangle(output, cv::Point(left,top), cv::Point(left + width,top + height), cv::Scalar(0,255,0),2);
        // Name


        std::string class_string = coconame[pred[i].label] + ' ' + std::to_string(pred[i].score).substr(0, 4);
        int baseline = 0;

        int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字體類型
        double fontScale = 1.0;              // 字體大小
        int thickness = 2;                   // 字體粗細
        cv::Size textSize = cv::getTextSize(class_string, fontFace, fontScale, thickness, &baseline);
        cv::Rect textRect(left, top- textSize.height, textSize.width, textSize.height);
        cv::rectangle(output,textRect,cv::Scalar(0,255,0),cv::FILLED);
        putText(output, class_string, cv::Point(left , top), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 1, 0);
    }
}
