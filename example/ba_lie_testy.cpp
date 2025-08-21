#include <bits/stdc++.h>
#include <math.h>
#include <iostream>

#include <Eigen/Core>
#include <camera/camera_model_pinhole_bal.hpp>
#include <bal_optmizer.hpp>

void print_summary(
    const OptResult& result, 
    const std::vector<Observation>& obsavations,
    const std::vector<Point3D>& points,
    const std::vector<CameraModelPinholeBal>& cameras,
    int camera_id
) 
{
    CameraModelPinholeBal camera0 = cameras[camera_id];
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "Camera ID: " << camera_id << std::endl;
    std::cout << "Residual: " << result.residual << std::endl;
    std::cout << "Optimized Rotation Matrix:\n" << result.R << std::endl;
    std::cout << "Optimized Translation Vector: " << result.t.transpose() << std::endl;
    std::cout << "Initial Rotation Matrix:\n" << camera0.rotationMatrix() << std::endl;
    std::cout << "Initial Translation Vector: " << camera0.get_translation().transpose() << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "R * R^T ? : \n" << result.R * result.R.transpose()  << std::endl;
    std::cout << " det(R) = 1 ?: det(R) = " << result.R.determinant() << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    int print_cnt = 0;
    for (Observation obs: obsavations) {
        if (print_cnt++ >= 5) break; // Limit to first 5 observations
        Eigen::Vector2d projected_point = camera0.project(points[obs.point_idx], 
                                                  result.R, 
                                                  result.t);
        std::cout << "Observation: Camera " << obs.camera_idx 
                  << ", Point " << obs.point_idx 
                  << ", 2D Point: (" << obs.x << ", " << obs.y << ")"
                  << ", Projected: (" << projected_point(0) << ", " << projected_point(1) << ")" << "\n";
    }
    std::cout << std::string(50, '=') << std::endl;
}
int main(int argc, char** argv) {
    int camera_id = (argc > 1) ? std::stoi(argv[1]) : 0; // Default to camera ID 0 if not provided

    BaOptimizer optimizer;
    optimizer.load_data("data/problem-16-22106-pre.txt");
    // auto opt_start_time = std::chrono::high_resolution_clock::now();
    // std::vector<OptResult> results = optimizer.optimize_all_cameras();
    // auto opt_end_time = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> opt_duration = opt_end_time - opt_start_time;
    // std::cout << "Optimization time: " << opt_duration.count() << " seconds" << std::endl;

    // ================================//
    // Print summary for first camera  //
    // ================================//
    std::vector<CameraModelPinholeBal> cameras = optimizer.get_cameras();
    std::vector<Observation> observations = optimizer.get_observations();
    std::vector<Point3D> points = optimizer.get_points();
    for (const auto& camera : cameras) {
        std::cout << "Camera ID: " << camera.get_camera_id() 
                  << ", Model: " << camera.get_camera_model_name()
                  << camera.rotationMatrix()
                  << std::endl;
    }
    CameraModelPinholeBal camera0 = cameras[camera_id];

    std::vector<Observation> obs0;
    for (const auto &obs : observations) {
        if (obs.camera_idx == camera0.get_camera_id()) {
            obs0.push_back(obs);
        }
    }
    std::vector<Point3D> points0;
    for (const auto &obs : obs0) {
        points0.push_back(points[obs.point_idx]);
    }
    for (const auto obs : obs0) {
        Eigen::Vector2d projected_point = camera0.project(points[obs.point_idx], 
                                                  camera0.rotationMatrix(), 
                                                  camera0.get_translation());
        std::cout << "Observation: Camera " << obs.camera_idx 
                  << ", Point " << obs.point_idx 
                  << ", 2D Point: (" << obs.x << ", " << obs.y << ")"
                  << ", Projected: (" << projected_point(0) << ", " << projected_point(1) << ")" << "\n";
        
    }
    OptResult result = optimizer.optimize_camera(obs0, camera0, points0);
    // OptResult result = results[camera_id];
    print_summary(result, obs0, points0, cameras, camera_id);
    

    return 0;
}

