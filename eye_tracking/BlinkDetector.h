#pragma once

#include <opencv2/opencv.hpp>

/**
 * @class BlinkDetector
 * @brief �� �������� �����Ͽ� Ŭ�� �̺�Ʈ�� �����ϴ� Ŭ�����Դϴ�.
 * ���� ���� ����(�����ڰ� ������ ����)�� ���� ������ �̻� ���ӵǴ��� �����մϴ�.
 */
class BlinkDetector {
public:
    /**
     * @brief BlinkDetector ������
     * @param required_frames �� ������ ���� ���� ���ܾ� ���������� �������� �����մϴ�.
     */
    BlinkDetector(int required_frames = 5);

    /**
     * @brief �� �����Ӹ��� ������ ���� ���θ� ������Ʈ�Ͽ� ������ ���¸� Ȯ���մϴ�.
     * @param is_pupil_detected ���� �����ӿ��� �����ڰ� �����Ǿ����� ���� (true/false)
     */
    void checkBlink(bool is_pupil_detected);

    /**
     * @brief ���� �������� �����Ǿ����� Ȯ���մϴ�.
     * @return �������� �����Ǿ����� true, �ƴϸ� false�� ��ȯ�մϴ�.
     */
    bool isBlinking();

    /**
     * @brief ������ ī���Ϳ� ���¸� �ʱ�ȭ�մϴ�. Ŭ�� ó�� �� ȣ���ؾ� �մϴ�.
     */
    void reset();

private:
    int blink_counter;                  // ���� ���� ���� ������ ���� ���� ī����
    int required_consecutive_frames;    // ���������� �����ϱ� ���� �ʿ��� ���� ������ ��
    bool blinking_state;                // ���� �������� �����Ǿ����� ���¸� �����ϴ� �÷���
};