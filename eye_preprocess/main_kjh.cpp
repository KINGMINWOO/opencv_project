#include <opencv2/opencv.hpp>
using namespace cv;

// ==============================
// 전처리 함수 (공통 모듈)
// ==============================
Mat preprocessEye(const Mat& eyeGray)
{
    Mat proc;

    // 1. Grayscale 변환 (혹시 입력이 컬러일 때 대비)
    if (eyeGray.channels() == 3)
        cvtColor(eyeGray, proc, COLOR_BGR2GRAY);
    else
        proc = eyeGray.clone();

    // 2. 히스토그램 평활화 (명암 대비 강화)
    equalizeHist(proc, proc);

    // 3. 블러링 (노이즈 줄이기)
    GaussianBlur(proc, proc, Size(5, 5), 0);

    // 4. 이진화 (Otsu 자동 임계값)
    threshold(proc, proc, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    // 5. Morphology (열림연산: 작은 잡음 제거)
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(proc, proc, MORPH_OPEN, kernel);

    // 6. 크기 정규화 (원하는 크기로 resize)
    resize(proc, proc, Size(60, 40));

    return proc; // 전처리 완료된 눈 이미지 반환
}
