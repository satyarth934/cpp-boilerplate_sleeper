/**
 Copyright 2017 by Satya Mallick ( Big Vision LLC )
 http://www.learnopencv.com
**/

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


void fillHoles(Mat &mask)
{
    /* 
     This hole filling algorithm is decribed in this post
     https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
     */
     
    Mat maskFloodfill = mask.clone();
    floodFill(maskFloodfill, cv::Point(0,0), Scalar(255));
    Mat mask2;
    bitwise_not(maskFloodfill, mask2);
    mask = (mask2 | mask);

}

int main(int argc, char** argv )
{
    // Read image
    Mat imgL = imread("/home/sandeep/Desktop/TerpBotics\ Dataset/leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png", 0);
    Mat imgR = imread("/home/sandeep/Desktop/TerpBotics\ Dataset/rightImg8bit_trainvaltest/rightImg8bit/train/aachen/aachen_000000_000019_rightImg8bit.png", 0);

    cv::Ptr<StereoBM> bm = cv::StereoBM::create(16,9);
    // bm->setROI1(roi1);
    // bm->setROI2(roi2);
    bm->setPreFilterCap(31);
    bm->setBlockSize(5);
    bm->setMinDisparity(0);
    bm->setNumDisparities(128);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);

    cv::Mat disp;
    bm->compute(imgL, imgR, disp);

    cv::imwrite("/home/sandeep/Desktop/disp.png", disp);


    // compute difference
    cv::Mat gt = cv::imread("/home/sandeep/Desktop/TerpBotics Dataset/disparity_trainvaltest/disparity/train/aachen/aachen_000000_000019_disparity.png", 0);
    cv::Mat diff;
    cv::Mat mask = cv::Mat::zeros(diff.size(), -1);
    cv::subtract(disp, gt, diff, mask, CV_8U);
    cv::imwrite("/home/sandeep/Desktop/diff.png", diff);
}
