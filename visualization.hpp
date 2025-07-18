#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "Base.hpp"  // For Detection struct

namespace robot {

// Visualization functions
void draw_detections(cv::Mat& image, const std::vector<Detection>& detections, 
                    const cv::Scalar& color = cv::Scalar(0, 255, 255), int thickness = 2);

void draw_landmarks(cv::Mat& image, const std::vector<cv::Point2f>& landmarks,
                   const std::vector<std::pair<int, int>>& connections,
                   const cv::Scalar& color = cv::Scalar(0, 255, 0), int thickness = 2, int radius = 3);

void draw_roi(cv::Mat& image, const std::vector<cv::Rect2f>& roi_boxes,
             const cv::Scalar& color = cv::Scalar(255, 0, 255), int thickness = 1);

// Hand landmark connections (21 points)
extern const std::vector<std::pair<int, int>> HAND_CONNECTIONS;

// Face landmark connections (simplified)
extern const std::vector<std::pair<int, int>> FACE_CONNECTIONS;

// Pose landmark connections
extern const std::vector<std::pair<int, int>> POSE_FULL_BODY_CONNECTIONS;
extern const std::vector<std::pair<int, int>> POSE_UPPER_BODY_CONNECTIONS;

} // namespace robot
