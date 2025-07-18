#ifndef GESTURE_HPP
#define GESTURE_HPP

#include "Detector.hpp"
#include <vector>
#include <string>
#include <opencv2/core.hpp>

namespace robot {

class Gesture {
public:
    Gesture(const std::string& model_name, std::shared_ptr<robot::HailoInference> inference_engine);
    ~Gesture() = default;

    void load_model(const std::string& model_path);
    std::string predict(const std::vector<cv::Point3f>& landmarks);
    
    // Add method to draw extended fingers visualization
    void draw_finger_status(cv::Mat& image, const cv::Point& position, int font_size = 1);
    
    // Get extended fingers status
    const std::vector<bool>& get_extended_fingers() const { return extended_fingers_; }

private:
    std::vector<float> extract_features(const std::vector<cv::Point3f>& landmarks);
    std::vector<bool> calculate_extended_fingers(const std::vector<cv::Point3f>& landmarks);

    std::string model_name_;
    std::shared_ptr<robot::HailoInference> inference_engine_;
    int hef_id_ = -1;
    std::vector<VStreamInfo> input_vstream_infos_;
    std::vector<VStreamInfo> output_vstream_infos_;
    std::vector<std::string> gesture_classes_ = {
        "fist", "open_hand", "point"
    };
    // Note: The actual model output order might be different from the class names order.
    // Based on the logs, the model seems to output in the order: [fist, open_hand, point]
    
    // Store extended fingers status
    std::vector<bool> extended_fingers_ = std::vector<bool>(5, false);
    
    // Finger names for visualization
    static const std::vector<std::string> finger_names_;
};
} // namespace robot

#endif // GESTURE_HPP
