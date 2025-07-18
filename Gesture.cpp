#include "Gesture.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace robot {

// Define finger names for logging
const std::vector<std::string> Gesture::finger_names_ = {"Thumb", "Index", "Middle", "Ring", "Pinky"};

std::vector<float> Gesture::extract_features(const std::vector<cv::Point3f>& landmarks) {
    if (landmarks.size() != 21) {
        throw std::runtime_error("Expected 21 landmarks, but got " + std::to_string(landmarks.size()));
    }

    std::vector<float> row;
    std::vector<std::array<float, 3>> coords;
    for (const auto& lm : landmarks) {
        row.push_back(lm.x);
        row.push_back(lm.y);
        row.push_back(lm.z);
        coords.push_back({lm.x, lm.y, lm.z});
    }

    // Pairwise distances (upper triangle, excluding diagonal)
    std::vector<float> dists;
    for (int i = 0; i < 21; ++i) {
        for (int j = i + 1; j < 21; ++j) {
            float dx = coords[i][0] - coords[j][0];
            float dy = coords[i][1] - coords[j][1];
            float dz = coords[i][2] - coords[j][2];
            dists.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
        }
    }

    // Angles for each finger (wrist, MCP, tip)
    std::vector<float> angles;
    std::vector<std::tuple<int, int, int>> finger_indices = {
        {0, 2, 4},   // Thumb
        {0, 5, 8},   // Index
        {0, 9, 12},  // Middle
        {0, 13, 16}, // Ring
        {0, 17, 20}  // Pinky
    };
    for (const auto& idx : finger_indices) {
        int a, b, c;
        std::tie(a, b, c) = idx;
        float ba[3] = {coords[a][0] - coords[b][0], coords[a][1] - coords[b][1], coords[a][2] - coords[b][2]};
        float bc[3] = {coords[c][0] - coords[b][0], coords[c][1] - coords[b][1], coords[c][2] - coords[b][2]};
        float dot = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2];
        float norm_ba = std::sqrt(ba[0]*ba[0] + ba[1]*ba[1] + ba[2]*ba[2]);
        float norm_bc = std::sqrt(bc[0]*bc[0] + bc[1]*bc[1] + bc[2]*bc[2]);
        float cosine_angle = dot / (norm_ba * norm_bc + 1e-8f);
        // Clamp to [-1, 1] to avoid NaNs
        cosine_angle = std::max(-1.0f, std::min(1.0f, cosine_angle));
        float angle = std::acos(cosine_angle);
        angles.push_back(angle);
    }

    // Calculate extended fingers for visualization
    extended_fingers_ = calculate_extended_fingers(landmarks);

    // Append features to row
    row.insert(row.end(), dists.begin(), dists.end());
    row.insert(row.end(), angles.begin(), angles.end());

    // (Optional: class filtering logic is not needed here, as it's for data cleaning in Python)

    return row;
}

std::vector<bool> Gesture::calculate_extended_fingers(const std::vector<cv::Point3f>& landmarks) {
    std::vector<bool> extended(5, false);  // Thumb, Index, Middle, Ring, Pinky
    
    if (landmarks.size() != 21) {
        return extended;
    }
    
    // First, normalize the landmarks to 0-1 range
    // Find the bounding box of the hand
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::min();
    float max_y = std::numeric_limits<float>::min();
    
    for (const auto& lm : landmarks) {
        min_x = std::min(min_x, lm.x);
        min_y = std::min(min_y, lm.y);
        max_x = std::max(max_x, lm.x);
        max_y = std::max(max_y, lm.y);
    }
    
    float width = max_x - min_x;
    float height = max_y - min_y;
    float size = std::max(width, height);
    
    if (size <= 0) {
        return extended;
    }
    
    // Create normalized landmarks
    std::vector<cv::Point3f> normalized_landmarks;
    for (const auto& lm : landmarks) {
        // Normalize to 0-1 range
        float norm_x = (lm.x - min_x) / size;
        float norm_y = (lm.y - min_y) / size;
        float norm_z = lm.z / size;  // Z is already relative
        normalized_landmarks.emplace_back(norm_x, norm_y, norm_z);
    }
    
    // Define key landmark indices
    const int WRIST = 0;
    const int THUMB_CMC = 1;
    const int THUMB_MCP = 2;
    const int THUMB_IP = 3;
    const int THUMB_TIP = 4;
    const int INDEX_MCP = 5;
    const int INDEX_PIP = 6;
    const int INDEX_DIP = 7;
    const int INDEX_TIP = 8;
    const int MIDDLE_MCP = 9;
    const int MIDDLE_PIP = 10;
    const int MIDDLE_DIP = 11;
    const int MIDDLE_TIP = 12;
    const int RING_MCP = 13;
    const int RING_PIP = 14;
    const int RING_DIP = 15;
    const int RING_TIP = 16;
    const int PINKY_MCP = 17;
    const int PINKY_PIP = 18;
    const int PINKY_DIP = 19;
    const int PINKY_TIP = 20;
    
    // Calculate angles between finger joints to determine if fingers are bent
    // For each finger, we'll check the angle between the joints
    
    // Helper function to calculate angle between three points
    auto calculate_angle = [](const cv::Point3f& a, const cv::Point3f& b, const cv::Point3f& c) -> float {
        float ba_x = a.x - b.x;
        float ba_y = a.y - b.y;
        float ba_z = a.z - b.z;
        
        float bc_x = c.x - b.x;
        float bc_y = c.y - b.y;
        float bc_z = c.z - b.z;
        
        float dot_product = ba_x * bc_x + ba_y * bc_y + ba_z * bc_z;
        float ba_mag = std::sqrt(ba_x * ba_x + ba_y * ba_y + ba_z * ba_z);
        float bc_mag = std::sqrt(bc_x * bc_x + bc_y * bc_y + bc_z * bc_z);
        
        if (ba_mag * bc_mag < 1e-8) {
            return 0.0f;
        }
        
        float cos_angle = dot_product / (ba_mag * bc_mag);
        cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));
        
        return std::acos(cos_angle) * 180.0f / M_PI;  // Convert to degrees
    };
    
    // Calculate finger extension using both distance and angle methods
    
    // 1. Distance-based method (MCP to tip)
    std::vector<std::pair<int, int>> mcp_tip_pairs = {
        {THUMB_MCP, THUMB_TIP},      // Thumb
        {INDEX_MCP, INDEX_TIP},      // Index
        {MIDDLE_MCP, MIDDLE_TIP},    // Middle
        {RING_MCP, RING_TIP},        // Ring
        {PINKY_MCP, PINKY_TIP}       // Pinky
    };
    
    std::vector<float> mcp_tip_distances;
    for (const auto& [mcp, tip] : mcp_tip_pairs) {
        const cv::Point3f& mcp_pt = normalized_landmarks[mcp];
        const cv::Point3f& tip_pt = normalized_landmarks[tip];
        float dx = tip_pt.x - mcp_pt.x;
        float dy = tip_pt.y - mcp_pt.y;
        float dz = tip_pt.z - mcp_pt.z;
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        mcp_tip_distances.push_back(dist);
    }
    
    // 2. Angle-based method (check if fingers are bent)
    // For better accuracy, we'll check angles at both PIP and DIP joints
    std::vector<std::vector<int>> finger_joint_triplets = {
        {THUMB_CMC, THUMB_MCP, THUMB_IP},      // Thumb CMC-MCP-IP
        {INDEX_MCP, INDEX_PIP, INDEX_DIP},     // Index MCP-PIP-DIP
        {MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP},  // Middle MCP-PIP-DIP
        {RING_MCP, RING_PIP, RING_DIP},        // Ring MCP-PIP-DIP
        {PINKY_MCP, PINKY_PIP, PINKY_DIP}      // Pinky MCP-PIP-DIP
    };
    
    // Additional angle checks at DIP-TIP joints
    std::vector<std::vector<int>> finger_tip_triplets = {
        {THUMB_MCP, THUMB_IP, THUMB_TIP},      // Thumb MCP-IP-TIP
        {INDEX_PIP, INDEX_DIP, INDEX_TIP},     // Index PIP-DIP-TIP
        {MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP},  // Middle PIP-DIP-TIP
        {RING_PIP, RING_DIP, RING_TIP},        // Ring PIP-DIP-TIP
        {PINKY_PIP, PINKY_DIP, PINKY_TIP}      // Pinky PIP-DIP-TIP
    };
    
    std::vector<float> joint_angles;
    for (const auto& triplet : finger_joint_triplets) {
        float angle = calculate_angle(
            normalized_landmarks[triplet[0]],
            normalized_landmarks[triplet[1]],
            normalized_landmarks[triplet[2]]
        );
        joint_angles.push_back(angle);
    }
    
    std::vector<float> tip_angles;
    for (const auto& triplet : finger_tip_triplets) {
        float angle = calculate_angle(
            normalized_landmarks[triplet[0]],
            normalized_landmarks[triplet[1]],
            normalized_landmarks[triplet[2]]
        );
        tip_angles.push_back(angle);
    }
    
    // 3. Calculate vertical position of fingertips relative to MCP joints
    // This helps distinguish between open hand and fist
    std::vector<float> vertical_positions;
    for (size_t i = 0; i < mcp_tip_pairs.size(); ++i) {
        int mcp = mcp_tip_pairs[i].first;
        int tip = mcp_tip_pairs[i].second;
        // Positive value means fingertip is above MCP (extended), negative means below (folded)
        float vertical_pos = normalized_landmarks[mcp].y - normalized_landmarks[tip].y;
        vertical_positions.push_back(vertical_pos);
    }
    
    // Log the distances and angles for debugging
    std::cout << "Finger distances: ";
    for (size_t i = 0; i < mcp_tip_distances.size(); ++i) {
        std::cout << finger_names_[i] << "=" << mcp_tip_distances[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Finger angles (MCP-PIP-DIP): ";
    for (size_t i = 0; i < joint_angles.size(); ++i) {
        std::cout << finger_names_[i] << "=" << joint_angles[i] << "° ";
    }
    std::cout << std::endl;
    
    std::cout << "Finger angles (PIP-DIP-TIP): ";
    for (size_t i = 0; i < tip_angles.size(); ++i) {
        std::cout << finger_names_[i] << "=" << tip_angles[i] << "° ";
    }
    std::cout << std::endl;
    
    std::cout << "Vertical positions: ";
    for (size_t i = 0; i < vertical_positions.size(); ++i) {
        std::cout << finger_names_[i] << "=" << vertical_positions[i] << " ";
    }
    std::cout << std::endl;
    
    // Calculate the average position of the palm (using wrist and MCP joints)
    cv::Point3f palm_center(0, 0, 0);
    palm_center.x = (normalized_landmarks[WRIST].x + 
                    normalized_landmarks[INDEX_MCP].x + 
                    normalized_landmarks[MIDDLE_MCP].x + 
                    normalized_landmarks[RING_MCP].x + 
                    normalized_landmarks[PINKY_MCP].x) / 5.0f;
    
    palm_center.y = (normalized_landmarks[WRIST].y + 
                    normalized_landmarks[INDEX_MCP].y + 
                    normalized_landmarks[MIDDLE_MCP].y + 
                    normalized_landmarks[RING_MCP].y + 
                    normalized_landmarks[PINKY_MCP].y) / 5.0f;
    
    palm_center.z = (normalized_landmarks[WRIST].z + 
                    normalized_landmarks[INDEX_MCP].z + 
                    normalized_landmarks[MIDDLE_MCP].z + 
                    normalized_landmarks[RING_MCP].z + 
                    normalized_landmarks[PINKY_MCP].z) / 5.0f;
    
    // Check if fingertips are close to the palm center (fist)
    std::vector<float> tip_to_palm_distances;
    std::vector<int> tip_indices = {THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP};
    
    for (int tip_idx : tip_indices) {
        float dx = normalized_landmarks[tip_idx].x - palm_center.x;
        float dy = normalized_landmarks[tip_idx].y - palm_center.y;
        float dz = normalized_landmarks[tip_idx].z - palm_center.z;
        float dist_to_palm = std::sqrt(dx*dx + dy*dy + dz*dz);
        tip_to_palm_distances.push_back(dist_to_palm);
    }
    
    std::cout << "Tip to palm distances: ";
    for (size_t i = 0; i < tip_to_palm_distances.size(); ++i) {
        std::cout << finger_names_[i] << "=" << tip_to_palm_distances[i] << " ";
    }
    std::cout << std::endl;
    
    // Improved finger extension detection using multiple metrics
    // Thresholds adjusted based on the debug output
    const float distance_threshold = 0.15f;       // MCP to tip distance threshold
    const float angle_threshold_low = 120.0f;     // Minimum angle for slightly bent fingers
    const float angle_threshold_high = 160.0f;    // Minimum angle for straight fingers
    const float vertical_threshold = 0.05f;       // Vertical position threshold
    const float palm_distance_threshold = 0.25f;  // Distance to palm threshold for fist detection
    
    // First check if we have a fist by checking if all fingertips are close to palm
    bool is_fist = true;
    for (size_t i = 0; i < tip_to_palm_distances.size(); ++i) {
        // Thumb has special handling
        if (i == 0) { // Thumb
            if (tip_to_palm_distances[i] > palm_distance_threshold * 1.2f) {
                is_fist = false;
            }
        } else {
            if (tip_to_palm_distances[i] > palm_distance_threshold) {
                is_fist = false;
            }
        }
    }
    
    if (is_fist) {
        std::cout << "Detected fist gesture: all fingertips close to palm" << std::endl;
        std::fill(extended.begin(), extended.end(), false);
        return extended;
    }
    
    // If not a fist, determine each finger's extension state
    for (size_t i = 0; i < 5; ++i) {
        // Special case for thumb which has different geometry
        if (i == 0) {  // Thumb
            // Thumb is extended if:
            // 1. Distance from MCP to tip is significant
            // 2. Angle at IP joint is relatively straight
            // 3. Tip is far from palm center
            bool distance_check = (mcp_tip_distances[i] > distance_threshold * 0.8f);
            bool angle_check = (joint_angles[i] > angle_threshold_low && tip_angles[i] > angle_threshold_low);
            bool palm_check = (tip_to_palm_distances[i] > palm_distance_threshold);
            
            extended[i] = distance_check && (angle_check || palm_check);
        } else {
            // For other fingers:
            // 1. Distance from MCP to tip must be significant
            // 2. Angles at both joints should be relatively straight
            // 3. Fingertip should be above MCP (vertical position positive)
            // 4. Tip should be far from palm center
            bool distance_check = (mcp_tip_distances[i] > distance_threshold);
            bool angle_check = (joint_angles[i] > angle_threshold_low && tip_angles[i] > angle_threshold_low);
            bool vertical_check = (vertical_positions[i] > vertical_threshold);
            bool palm_check = (tip_to_palm_distances[i] > palm_distance_threshold);
            
            // For index finger specifically, be more strict
            if (i == 1) {  // Index finger
                extended[i] = distance_check && angle_check && (vertical_check || palm_check);
            } 
            // For middle, ring, and pinky
            else {
                extended[i] = distance_check && (angle_check || vertical_check) && palm_check;
            }
        }
    }
    
    // Double-check for common misclassification patterns
    
    // Check for fist with slightly extended fingers
    // In a fist, fingertips should be below MCP joints (negative vertical position)
    bool all_fingers_down = true;
    for (size_t i = 1; i < vertical_positions.size(); i++) {  // Skip thumb
        if (vertical_positions[i] > 0) {
            all_fingers_down = false;
            break;
        }
    }
    
    // If all fingers are pointing down and we detected some extensions, it's likely a misclassification
    if (all_fingers_down && (extended[1] || extended[2] || extended[3] || extended[4])) {
        std::cout << "Correcting misclassification: fingers are pointing down, likely a fist" << std::endl;
        std::fill(extended.begin() + 1, extended.end(), false);  // Set all fingers except thumb to not extended
    }
    
    // Check for open hand with some fingers not detected as extended
    // In an open hand, most fingertips should be far from palm
    int far_from_palm_count = 0;
    for (size_t i = 0; i < tip_to_palm_distances.size(); i++) {
        if (tip_to_palm_distances[i] > palm_distance_threshold * 1.2) {
            far_from_palm_count++;
        }
    }
    
    // If 4-5 fingertips are far from palm but we didn't detect them all as extended,
    // it might be an open hand with some misclassification
    if (far_from_palm_count >= 4 && (extended[1] || extended[2] || extended[3] || extended[4])) {
        int extended_count = 0;
        for (bool e : extended) if (e) extended_count++;
        
        if (extended_count < 4) {
            std::cout << "Possible open hand detected based on palm distances, but only " 
                      << extended_count << " fingers classified as extended" << std::endl;
            
            // Don't automatically correct this case, as it might be a legitimate partial extension
        }
    }
    
    // Log the extended fingers
    std::cout << "Extended fingers: ";
    int count = 0;
    for (size_t i = 0; i < extended.size(); ++i) {
        if (extended[i]) {
            count++;
            std::cout << finger_names_[i] << " ";
        }
    }
    std::cout << "(" << count << " total)" << std::endl;
    
    return extended;
}


Gesture::Gesture(const std::string& model_name, std::shared_ptr<robot::HailoInference> inference_engine)
    : model_name_(model_name), inference_engine_(inference_engine) {}

void Gesture::load_model(const std::string& model_path) {
    hef_id_ = inference_engine_->load_model(model_path);
    input_vstream_infos_ = inference_engine_->get_input_vstream_infos(hef_id_);
    output_vstream_infos_ = inference_engine_->get_output_vstream_infos(hef_id_);
}

std::string Gesture::predict(const std::vector<cv::Point3f>& landmarks) {
    if (landmarks.empty()) {
        std::cout << "Gesture::predict - No landmarks provided" << std::endl;
        return "";
    }

    std::cout << "Gesture::predict - Processing " << landmarks.size() << " landmarks" << std::endl;
    
    // Reset extended fingers
    extended_fingers_ = std::vector<bool>(5, false);

    try {
        std::vector<float> features = extract_features(landmarks);
        std::cout << "Gesture::predict - Extracted " << features.size() << " features" << std::endl;
        
        // The model expects a batch of inputs
        cv::Mat features_mat(1, features.size(), CV_32F, features.data());
        
        std::map<std::string, cv::Mat> input_data;
        input_data[input_vstream_infos_[0].name] = features_mat;

        auto results = inference_engine_->infer(input_data, hef_id_);
        std::cout << "Gesture::predict - Got " << results.size() << " inference results" << std::endl;
        
        if (results.empty()) {
            std::cout << "Gesture::predict - No inference results" << std::endl;
            return "";
        }

        // Get the output matrix
        const auto& output_mat = results.begin()->second;
        
        // Debug the output matrix in detail
        std::cout << "Gesture::predict - Output matrix type: " << output_mat.type() << std::endl;
        std::cout << "Gesture::predict - Output matrix dims: " << output_mat.dims << std::endl;
        std::cout << "Gesture::predict - Output shape: " << output_mat.rows << "x" << output_mat.cols << "x" << output_mat.channels() << std::endl;
        
        // Try to access the data in different ways based on the shape
        int num_classes = 0;
        std::vector<float> class_scores;
        
        if (output_mat.dims == 3 || output_mat.channels() == 3) {
            // The output is a 3D tensor with shape (1, 1, 3)
            num_classes = 3;
            class_scores.resize(num_classes);
            
            if (output_mat.channels() == 3) {
                // Access as a multi-channel 2D matrix
                for (int i = 0; i < num_classes; ++i) {
                    class_scores[i] = output_mat.at<cv::Vec3f>(0, 0)[i];
                }
            } else {
                // Try to access as a 3D tensor
                for (int i = 0; i < num_classes; ++i) {
                    class_scores[i] = output_mat.at<float>(0, 0, i);
                }
            }
        } else if (output_mat.cols == 3) {
            // The output is a 2D matrix with shape (1, 3)
            num_classes = output_mat.cols;
            const float* output_scores = output_mat.ptr<float>(0);
            class_scores.assign(output_scores, output_scores + num_classes);
        } else {
            // Fallback to single value
            num_classes = 1;
            class_scores.push_back(output_mat.at<float>(0, 0));
        }
        
        std::cout << "Gesture::predict - Detected " << num_classes << " classes" << std::endl;
        std::cout << "Gesture::predict - Raw output scores: ";
        for (int i = 0; i < class_scores.size(); ++i) {
            std::cout << class_scores[i] << " ";
        }
        std::cout << std::endl;

        // Process the model outputs
        int predicted_index = 0;
        float max_score = 0.0f;
        
        if (num_classes == 3) {
            // The model is correctly outputting 3 class probabilities
            std::vector<float> probabilities = class_scores;
            
            // The model might output raw logits, so apply softmax if needed
            float max_val = *std::max_element(probabilities.begin(), probabilities.end());
            float sum_exp = 0.0f;
            for (size_t i = 0; i < probabilities.size(); ++i) {
                probabilities[i] = std::exp(probabilities[i] - max_val);
                sum_exp += probabilities[i];
            }
            if (sum_exp > 0) {
                for (size_t i = 0; i < probabilities.size(); ++i) {
                    probabilities[i] /= sum_exp;
                }
            }

            std::cout << "Gesture::predict - Softmax probabilities: ";
            for (size_t i = 0; i < probabilities.size(); ++i) {
                std::cout << probabilities[i] << " ";
            }
            std::cout << std::endl;

            auto max_it = std::max_element(probabilities.begin(), probabilities.end());
            predicted_index = std::distance(probabilities.begin(), max_it);
            max_score = *max_it;
            
            std::cout << "Gesture::predict - Multi-class model, highest probability at index: " 
                      << predicted_index << " with score: " << max_score << std::endl;
        } else if (num_classes == 1) {
            // Fallback for single-output model
            float score = class_scores[0];
            
            // Let's adjust our thresholds to improve open_hand detection:
            // 0.0-0.25: fist (index 0)
            // 0.25-0.55: open_hand (index 1)
            // 0.55-1.0: point (index 2)
            
            // Log the raw score for debugging
            std::cout << "Gesture::predict - Raw score for gesture mapping: " << score << std::endl;
            
            if (score < 0.25f) {
                predicted_index = 0; // fist
            } else if (score < 0.55f) {
                predicted_index = 1; // open_hand
            } else {
                predicted_index = 2; // point
            }
            
            max_score = 1.0f;
            std::cout << "Gesture::predict - Single output model, score: " << score 
                      << ", mapped to index: " << predicted_index << std::endl;
        } else {
            // Unexpected number of outputs
            std::cout << "Gesture::predict - Unexpected number of outputs: " << num_classes << std::endl;
            
            // Default to the first class
            predicted_index = 0;
            max_score = 1.0f;
        }

        std::cout << "Gesture::predict - Predicted index: " << predicted_index << " with score: " << max_score << std::endl;

        // Based on the logs, it appears the model's output order doesn't match our class order
        // The model seems to be outputting: [fist, open_hand, point] but recognizing open_hand as fist
        // Let's map the model's output to our gesture classes more accurately
        
        // Count extended fingers for better decision making
        int extended_count = 0;
        for (bool extended : extended_fingers_) {
            if (extended) extended_count++;
        }
        
        std::cout << "Gesture::predict - Extended finger count: " << extended_count << std::endl;
        
        // Use finger extension as the primary classification method
        // This is more reliable than the model output in many cases
        
        // If 4 or 5 fingers are extended, it's an open hand
        if (extended_count >= 4) {
            std::cout << "Gesture::predict - Many extended fingers (4-5), classifying as open_hand" << std::endl;
            std::string result = "open_hand";
            std::cout << "Gesture::predict - Returning: " << result << std::endl;
            return result;
        }
        
        // If only the index finger is extended, it's a point gesture
        if ((extended_count == 1 && extended_fingers_[1]) ||  // Only index finger extended
            (extended_count == 2 && extended_fingers_[1] && extended_fingers_[0])) {  // Index + thumb extended
            std::cout << "Gesture::predict - Index finger extended (possibly with thumb), classifying as point" << std::endl;
            std::string result = "point";
            std::cout << "Gesture::predict - Returning: " << result << std::endl;
            return result;
        }
        
        // If no fingers are extended, it's a fist
        if (extended_count == 0 || (extended_count == 1 && extended_fingers_[0])) {  // No fingers or only thumb
            std::cout << "Gesture::predict - No fingers extended (or only thumb), classifying as fist" << std::endl;
            std::string result = "fist";
            std::cout << "Gesture::predict - Returning: " << result << std::endl;
            return result;
        }
        
        // For cases where 2-3 fingers are extended, or only non-index fingers are extended,
        // use a more sophisticated decision logic
        
        // If middle, ring, and pinky are extended (but not index), it's likely a specific gesture
        // but we'll classify as fist for simplicity
        if (!extended_fingers_[1] && (extended_fingers_[2] || extended_fingers_[3] || extended_fingers_[4])) {
            std::cout << "Gesture::predict - Non-index fingers extended, classifying as fist" << std::endl;
            std::string result = "fist";
            std::cout << "Gesture::predict - Returning: " << result << std::endl;
            return result;
        }
        
        // For other ambiguous cases, fall back to the model prediction
        std::cout << "Gesture::predict - Using model prediction for ambiguous finger extension pattern" << std::endl;
        
        if (predicted_index < (int)gesture_classes_.size()) {
            std::string result = gesture_classes_[predicted_index];
            std::cout << "Gesture::predict - Returning: " << result << std::endl;
            return result;
        } else {
            std::cout << "Gesture::predict - Predicted index out of bounds" << std::endl;
        }

    } catch (const std::exception& e) {
        // Handle errors, e.g., if feature extraction fails
        std::cout << "Gesture::predict - Exception: " << e.what() << std::endl;
        return "";
    }

    return "";
}

void Gesture::draw_finger_status(cv::Mat& image, const cv::Point& position, int font_size) {
    if (image.empty() || extended_fingers_.empty()) {
        return;
    }
    
    const int line_height = 30 * font_size;
    const float font_scale = 0.6f * font_size;
    const int thickness = 1 * font_size;
    
    // Draw title
    cv::putText(image, "Extended Fingers:", 
                cv::Point(position.x, position.y), 
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
    
    // Count extended fingers
    int extended_count = 0;
    for (bool extended : extended_fingers_) {
        if (extended) extended_count++;
    }
    
    // Draw count
    std::string count_text = "Count: " + std::to_string(extended_count) + "/5";
    cv::putText(image, count_text, 
                cv::Point(position.x, position.y + line_height), 
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
    
    // Draw individual finger status
    for (size_t i = 0; i < extended_fingers_.size(); ++i) {
        std::string finger_text = finger_names_[i] + ": " + (extended_fingers_[i] ? "Extended" : "Folded");
        cv::Scalar color = extended_fingers_[i] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::putText(image, finger_text, 
                    cv::Point(position.x, position.y + (i + 2) * line_height), 
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness);
    }
}

} // namespace robot
