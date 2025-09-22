// preprocess.cpp
#include "preprocess.h"

cv::Mat preprocessEye(const cv::Mat& eyeGray)
{
    cv::Mat proc;

    // 1. Grayscale 변환
    if (eyeGray.channels() == 3)
        cv::cvtColor(eyeGray, proc, cv::COLOR_BGR2GRAY);
    else
        proc = eyeGray.clone();

    // 2. 히스토그램 평활화
    cv::equalizeHist(proc, proc);

    // 3. 가우시안 블러
    cv::GaussianBlur(proc, proc, cv::Size(5, 5), 0);

    // 4. Otsu 자동 이진화
    cv::threshold(proc, proc, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 5. Morphology (열림 연산: 작은 잡음 제거)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(proc, proc, cv::MORPH_OPEN, kernel);

    // 6. 크기 정규화
    cv::resize(proc, proc, cv::Size(60, 40));

    return proc;
}
