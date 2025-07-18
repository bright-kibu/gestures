#include "visualization.hpp"
#include <iostream>

namespace robot {

// Hand landmark connections (21 points - MediaPipe format)
const std::vector<std::pair<int, int>> HAND_CONNECTIONS = {
    // Thumb
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    // Index finger
    {0, 5}, {5, 6}, {6, 7}, {7, 8},
    // Middle finger
    {0, 9}, {9, 10}, {10, 11}, {11, 12},
    // Ring finger
    {0, 13}, {13, 14}, {14, 15}, {15, 16},
    // Pinky finger
    {0, 17}, {17, 18}, {18, 19}, {19, 20}
};

// Face landmark connections (simplified - key facial features)
const std::vector<std::pair<int, int>> FACE_CONNECTIONS = {
    // Face outline (simplified)
    {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8},
    {8, 9}, {9, 10}, {10, 11}, {11, 12}, {12, 13}, {13, 14}, {14, 15}, {15, 16},
    // Eyebrows
    {17, 18}, {18, 19}, {19, 20}, {20, 21},
    {22, 23}, {23, 24}, {24, 25}, {25, 26},
    // Eyes
    {36, 37}, {37, 38}, {38, 39}, {39, 40}, {40, 41}, {41, 36},
    {42, 43}, {43, 44}, {44, 45}, {45, 46}, {46, 47}, {47, 42},
    // Nose
    {27, 28}, {28, 29}, {29, 30}, {31, 32}, {32, 33}, {33, 34}, {34, 35},
    // Mouth
    {48, 49}, {49, 50}, {50, 51}, {51, 52}, {52, 53}, {53, 54}, {54, 55}, {55, 56},
    {56, 57}, {57, 58}, {58, 59}, {59, 48}
};

// Pose landmark connections - upper body (simplified)
const std::vector<std::pair<int, int>> POSE_UPPER_BODY_CONNECTIONS = {
    // Head and shoulders
    {0, 1}, {1, 2}, {2, 3}, {3, 7},
    {0, 4}, {4, 5}, {5, 6}, {6, 8},
    // Torso
    {9, 10}, {11, 12}, {11, 13}, {13, 15},
    {12, 14}, {14, 16}, {11, 23}, {12, 24}, {23, 24}
};

// Pose landmark connections - full body
const std::vector<std::pair<int, int>> POSE_FULL_BODY_CONNECTIONS = {
    // Head and shoulders
    {0, 1}, {1, 2}, {2, 3}, {3, 7},
    {0, 4}, {4, 5}, {5, 6}, {6, 8},
    // Torso
    {9, 10}, {11, 12}, {11, 13}, {13, 15},
    {12, 14}, {14, 16}, {11, 23}, {12, 24}, {23, 24},
    // Legs
    {23, 25}, {25, 27}, {27, 29}, {29, 31},
    {24, 26}, {26, 28}, {28, 30}, {30, 32}
};

void draw_detections(cv::Mat& image, const std::vector<Detection>& detections, 
                    const cv::Scalar& color, int thickness) {
    for (const auto& detection : detections) {
        // Create bounding box from detection coordinates
        cv::Rect bbox(detection.xmin, detection.ymin, 
                     detection.xmax - detection.xmin, 
                     detection.ymax - detection.ymin);
        
        // Draw bounding box
        cv::rectangle(image, bbox, color, thickness);
        
        // Draw score if confidence is reasonable
        if (detection.score > 0.0f) {
            std::string score_text = std::to_string(detection.score).substr(0, 4);
            cv::Point text_pos(bbox.x, bbox.y - 5);
            cv::putText(image, score_text, text_pos, 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
    }
}

void draw_landmarks(cv::Mat& image, const std::vector<cv::Point2f>& landmarks,
                   const std::vector<std::pair<int, int>>& connections,
                   const cv::Scalar& color, int thickness, int radius) {
    
    // Draw landmark points
    for (const auto& point : landmarks) {
        if (point.x >= 0 && point.y >= 0 && 
            point.x < image.cols && point.y < image.rows) {
            cv::circle(image, point, radius, color, -1);
        }
    }
    
    // Draw connections
    for (const auto& connection : connections) {
        int idx1 = connection.first;
        int idx2 = connection.second;
        
        if (idx1 >= 0 && idx1 < static_cast<int>(landmarks.size()) &&
            idx2 >= 0 && idx2 < static_cast<int>(landmarks.size())) {
            
            const cv::Point2f& p1 = landmarks[idx1];
            const cv::Point2f& p2 = landmarks[idx2];
            
            // Check if points are valid
            if (p1.x >= 0 && p1.y >= 0 && p1.x < image.cols && p1.y < image.rows &&
                p2.x >= 0 && p2.y >= 0 && p2.x < image.cols && p2.y < image.rows) {
                cv::line(image, p1, p2, color, thickness);
            }
        }
    }
}

void draw_roi(cv::Mat& image, const std::vector<cv::Rect2f>& roi_boxes,
             const cv::Scalar& color, int thickness) {
    for (const auto& roi : roi_boxes) {
        cv::rectangle(image, roi, color, thickness);
    }
}

} // namespace robot
