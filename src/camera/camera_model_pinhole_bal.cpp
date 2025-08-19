#include <iostream>
#include <filesystem>
#include <fstream>   // std::ifstream 用
#include <sstream>   // std::istringstream 用

#include <opencv2/core.hpp>
#include <Eigen/Geometry>
#include "camera/camera_model_pinhole_bal.hpp"

Eigen::Matrix<double, 2, 4> CameraModelPinholeBal::get_prj_jacobian(
    const Eigen::Vector3d& point
) const
{
    Eigen::Matrix<double, 2, 4> jacobian = Eigen::Matrix<double, 2, 4>::Zero();
    double x = point(0);
    double y = point(1);
    double z = point(2);

    jacobian(0, 0) = -f_ / z;
    jacobian(0, 2) = f_ * x / (z * z);
    jacobian(1, 1) = -f_ / z;
    jacobian(1, 2) = f_ * y / (z * z);
    return jacobian;
}

Eigen::Vector2d CameraModelPinholeBal::project(
    const Eigen::Vector3d &xyz,
    const Eigen::Matrix3d &R,
    const Eigen::Vector3d &t
) const
{
    Eigen::Vector3d P = R * xyz + t;

    Eigen::Vector2d p = -P.head<2>() / P.z();
    double r2 = p.squaredNorm();
    //double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
    //Eigen::Vector2d pp = f * radial * p;
    Eigen::Vector2d pp = f_ * p;

    return pp;
}

Eigen::Matrix3d CameraModelPinholeBal::rotationMatrix() const 
{
    double theta = rot_.norm();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (theta > 1e-12) {
        Eigen::Vector3d r = rot_ / theta;
        Eigen::AngleAxisd aa(theta, r);
        R = aa.toRotationMatrix();
    }
    return R;
}