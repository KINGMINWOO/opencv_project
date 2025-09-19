//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/objdetect.hpp>
//
//int main() {
//    // Load the pre-trained Haar cascade classifier for face detection
//    cv::CascadeClassifier face, eyes;
//    if (!face.load("haarcascade_frontalface_default.xml")) {
//        // Handle error if classifier not loaded
//        return -1;
//    }
//    if (!eyes.load("haarcascade_eye.xml")) {
//        // Handle error if classifier not loaded
//        return -1;
//    }
//
//    // Open the default camera
//    cv::VideoCapture cap(0);
//    if (!cap.isOpened()) {
//        // Handle error if camera not opened
//        return -1;
//    }
//
//    cv::Mat frame;
//    while (true) {
//        cap >> frame; // Read a new frame from the camera
//        if (frame.empty()) {
//            break;
//        }
//
//        cv::Mat gray_frame;
//        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY); // Convert to grayscale
//        cv::equalizeHist(gray_frame, gray_frame); // Equalize histogram for better detection
//
//        std::vector<cv::Rect> faceRect, eyesRect;
//        // detectMultiScale(입력 이미지, 추출된 영역, 이미지 축소 정도, 도형 이웃 수, 플래그, 추출 영역 최소 크기, 최대 크기)
//        if(!face.empty()) face.detectMultiScale(gray_frame, faceRect, 1.1, 8, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
//        if (!eyes.empty()) eyes.detectMultiScale(gray_frame, eyesRect, 1.1, 4, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
//
//        // Draw rectangles around detected things
//        for (const auto& face : faceRect) {
//            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
//        }
//        for (const auto& eyes : eyesRect) {
//            cv::rectangle(frame, eyes, cv::Scalar(0, 255, 0), 2);
//        }
//
//        cv::imshow("Face Detection", frame);
//        if (cv::waitKey(1) == 'q') { // Press 'q' to quit
//            break;
//        }
//    }
//
//    cap.release();
//    cv::destroyAllWindows();
//    return 0;
//}

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

int main() {
    // Haar Cascade 분류기 로드
    cv::CascadeClassifier face_cascade, eyes_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Error: face cascade not loaded." << std::endl;
        return -1;
    }
    if (!eyes_cascade.load("haarcascade_eye.xml")) {
        std::cerr << "Error: eyes cascade not loaded." << std::endl;
        return -1;
    }

    // 기본 카메라 열기
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: camera not opened." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray_frame, gray_frame);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray_frame, faces, 1.1, 10, 0, cv::Size(30, 30));

        // 각 얼굴에 대해 반복
        for (const auto& face : faces) {

            // 얼굴 영역을 ROI로 설정
            cv::Mat face_roi_gray = gray_frame(face);
            cv::Mat face_roi_color = frame(face);

            std::vector<cv::Rect> eyes;
            // 얼굴 안에서 눈 검출
            eyes_cascade.detectMultiScale(face_roi_gray, eyes, 1.1, 10, 0, cv::Size(20, 20));

            // 각 눈에 대해 반복
            if (!eyes_cascade.empty())
            {
                cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
                for (const auto& eye : eyes) {
                    cv::rectangle(face_roi_color, eye, cv::Scalar(255, 0, 0), 2);

                    // 눈 영역을 ROI로 설정하여 눈동자 찾기
                    cv::Mat eye_roi = face_roi_gray(eye);

                    // 1. 이진화 (Thresholding)
                    cv::Mat binary_eye;
                    // 임계값은 조명 환경에 따라 조절해야 할 수 있습니다.
                    cv::threshold(eye_roi, binary_eye, 80, 255, cv::THRESH_BINARY_INV);

                    // 2. 모폴로지 연산으로 노이즈 제거
                    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
                    cv::erode(binary_eye, binary_eye, kernel, cv::Point(-1, -1), 1);
                    cv::dilate(binary_eye, binary_eye, kernel, cv::Point(-1, -1), 2);

                    // 3. 윤곽선 찾기
                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(binary_eye, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                    double max_area = 0;
                    cv::Point pupil_center;
                    bool pupil_found = false;

                    // 4. 가장 큰 윤곽선을 찾아 중심점 계산
                    for (const auto& contour : contours) {
                        double area = cv::contourArea(contour);
                        if (area > max_area) {
                            max_area = area;
                            cv::Moments mu = cv::moments(contour);
                            // 모멘트를 이용해 중심점 계산
                            pupil_center = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
                            pupil_found = true;
                        }
                    }

                    if (pupil_found) {
                        // 원본 프레임 좌표계로 변환하여 원 그리기
                        cv::circle(face_roi_color, cv::Point(eye.x + pupil_center.x, eye.y + pupil_center.y), 3, cv::Scalar(0, 0, 255), -1);
                    }
                }
            }
        }

        cv::imshow("Pupil Detection", frame);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}