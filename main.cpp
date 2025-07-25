#include <iostream>
#include "DetectLiveCamera.hpp"

int main() {
    try {
        std::cout << "Simple Camera Display Test" << std::endl;
        std::cout << "Press 'q' or Ctrl+C to exit" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Starting camera..." << std::endl;

        robot::DetectLiveCamera camera;
        camera.run_camera_display_loop(1536, 864);

        std::cout << "Camera stopped." << std::endl;
        std::cout << "============================================" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An unhandled exception occurred: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown exception occurred." << std::endl;
        return 1;
    }

    return 0;
}
