// BlinkDetector.cpp

#include "BlinkDetector.h"

// 생성자: 필요한 연속 프레임 수를 초기화합니다.
BlinkDetector::BlinkDetector(int required_frames) {
    required_consecutive_frames = required_frames;
    reset(); // 모든 상태 변수를 초기 상태로 설정
}

// 매 프레임 눈동자 감지 결과를 기반으로 상태를 업데이트하는 함수
void BlinkDetector::checkBlink(bool is_pupil_detected) {
    if (!is_pupil_detected) {
        // 눈동자가 감지되지 않으면 카운터를 1 증가시킵니다.
        blink_counter++;
    }
    else {
        // 눈동자가 다시 감지되면, 연속성이 깨졌으므로 카운터를 0으로 리셋합니다.
        blink_counter = 0;
    }

    // 카운터가 설정된 프레임 수에 도달했고, 아직 깜빡임 상태가 아니라면
    if (blink_counter >= required_consecutive_frames && !blinking_state) {
        // 깜빡임 상태를 true로 설정합니다.
        blinking_state = true;
    }
}

// 현재 깜빡임 상태를 반환하는 함수
bool BlinkDetector::isBlinking() {
    return blinking_state;
}

// 모든 상태를 초기화하는 함수. 클릭 이벤트 처리 후 반드시 호출해야 합니다.
void BlinkDetector::reset() {
    blink_counter = 0;
    blinking_state = false;
}