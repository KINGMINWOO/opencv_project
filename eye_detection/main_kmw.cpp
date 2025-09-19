#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using std::cout; using std::endl;

static Point2f emaPoint(const Point2f& prev, const Point2f& cur, float alpha = 0.25f) {
    return prev * (1.0f - alpha) + cur * alpha;
}

static bool findPupil(const Mat& eyeGray, Point& pupil, float& radius)
{
    // 1) 전처리
    Mat blurImg; GaussianBlur(eyeGray, blurImg, Size(7, 7), 0);
    // 눈꺼풀/하이라이트 제거를 위해 상위 톤 억제
    Mat eq; equalizeHist(blurImg, eq);
    // 2) 동공은 어두움: Otsu + 반전
    Mat bin;
    threshold(eq, bin, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    // 3) 열림 연산으로 잡티 제거
    morphologyEx(bin, bin, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

    // 4) 큰 컨투어 중심을 후보로
    std::vector<std::vector<Point>> contours;
    findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        // 실패하면 허프원 시도
        std::vector<Vec3f> circles;
        HoughCircles(blurImg, circles, HOUGH_GRADIENT, 1, eyeGray.rows / 8, 200, 15, eyeGray.rows / 16, eyeGray.rows / 3);
        if (circles.empty()) return false;
        Vec3f c = circles[0];
        pupil = Point(cvRound(c[0]), cvRound(c[1]));
        radius = c[2];
        return true;
    }
    // 가장 큰 컨투어 선택
    size_t idxMax = 0; double maxA = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double a = contourArea(contours[i]);
        if (a > maxA) { maxA = a; idxMax = i; }
    }
    Point2f c; float r;
    minEnclosingCircle(contours[idxMax], c, r);
    pupil = Point(cvRound(c.x), cvRound(c.y));
    radius = r;
    return true;
}

int main()
{
    // 0) 분류기 로드 (OpenCV 설치 경로의 haarcascade 파일 경로를 맞춰주세요)
    // 예: Linux: /usr/share/opencv4/haarcascades/..., Windows: <opencv>/build/etc/haarcascades/...
    std::string face_cascade_path = "haarcascade_frontalface_default.xml";
    std::string eye_cascade_path = "haarcascade_eye_tree_eyeglasses.xml";

    CascadeClassifier faceCasc, eyeCasc;
    if (!faceCasc.load(face_cascade_path) || !eyeCasc.load(eye_cascade_path)) {
        std::cerr << "Failed to load cascades. Check paths.\n";
        return -1;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera\n";
        return -1;
    }
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    Point2f emaLeft(-1, -1), emaRight(-1, -1); // EMA 초기화
    while (true) {
        Mat frame; cap >> frame;
        if (frame.empty()) break;

        Mat gray; cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 1) 얼굴 검출
        std::vector<Rect> faces;
        faceCasc.detectMultiScale(gray, faces, 1.1, 3, 0, Size(120, 120));
        for (const Rect& f : faces) {
            rectangle(frame, f, Scalar(0, 255, 0), 2);

            // 2) 얼굴 ROI에서 눈 검출 (상반부 우선)
            Rect upperFace = Rect(f.x, f.y, f.width, f.height * 0.6);
            upperFace &= Rect(0, 0, frame.cols, frame.rows);
            Mat faceROI = gray(upperFace);

            std::vector<Rect> eyes;
            eyeCasc.detectMultiScale(faceROI, eyes, 1.1, 2, 0, Size(30, 30));
            // 좌/우 눈을 정렬해 사용
            std::sort(eyes.begin(), eyes.end(), [](const Rect& a, const Rect& b) { return a.x < b.x; });

            for (size_t i = 0; i < eyes.size() && i < 2; ++i) {
                Rect e = eyes[i];
                Rect eyeRect(e.x + upperFace.x, e.y + upperFace.y, e.width, e.height);
                rectangle(frame, eyeRect, Scalar(255, 200, 0), 2);

                Mat eyeGray = gray(eyeRect).clone();

                // 3) 동공 찾기
                Point pupil; float r = 0;
                bool ok = findPupil(eyeGray, pupil, r);
                if (ok) {
                    // 4) 프레임 좌표로 변환
                    Point pupilInFrame = Point(eyeRect.x + pupil.x, eyeRect.y + pupil.y);
                    circle(frame, pupilInFrame, (int)std::max(2.f, r), Scalar(0, 0, 255), 2);

                    // 시선 좌표 정규화 (-1..1): 눈 ROI 중심 기준
                    Point2f center(eyeRect.x + eyeRect.width * 0.5f, eyeRect.y + eyeRect.height * 0.5f);
                    Point2f offset = Point2f(pupilInFrame) - center;
                    Point2f norm(offset.x / (eyeRect.width * 0.5f),
                        offset.y / (eyeRect.height * 0.5f));

                    // 5) 흔들림 감소(EMA)
                    if (i == 0) { // 왼쪽 눈(화면상 좌측)
                        if (emaLeft.x < -0.5f) emaLeft = norm;
                        else emaLeft = emaPoint(emaLeft, norm, 0.2f);
                        putText(frame, cv::format("L(%.2f, %.2f)", emaLeft.x, emaLeft.y),
                            Point(eyeRect.x, eyeRect.y - 8), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
                    }
                    else {
                        if (emaRight.x < -0.5f) emaRight = norm;
                        else emaRight = emaPoint(emaRight, norm, 0.2f);
                        putText(frame, cv::format("R(%.2f, %.2f)", emaRight.x, emaRight.y),
                            Point(eyeRect.x, eyeRect.y - 8), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
                    }
                }
                else {
                    putText(frame, "pupil?", Point(eyeRect.x, eyeRect.y - 8),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 50, 255), 1);
                }
            }
        }

        // 안내 텍스트
        putText(frame, "Press 'q' to quit", Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        imshow("Eye Tracker (OpenCV)", frame);
        char key = (char)waitKey(1);
        if (key == 'q' || key == 27) break;
    }
    return 0;
}
