#include <bits/stdc++.h>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <camera/camera_model_pinhole_bal.hpp>

using Point3D = Eigen::Vector3d;

struct Observation {
  int camera_idx;
  int point_idx;
  double x, y;
};

struct OptResult {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    double residual;
};

class BaOptimizer 
{
public:
    BaOptimizer() = default;
    ~BaOptimizer() = default;

    std::vector<OptResult> optimize();

    OptResult optimize_camera(
        const std::vector<Observation> obs,
        const CameraModelPinholeBal cam,
        const std::vector<Point3D> points
    );

    void load_data(std::string path);

    std::vector<CameraModelPinholeBal> get_cameras()
    {
        return cameras_;
    }
    std::vector<Observation> get_observations()
    {
        return observations_;
    }
    std::vector<Point3D> get_points()
    {
        return points_;
    }

private:
    std::vector<CameraModelPinholeBal> cameras_;
    std::vector<Observation> observations_;
    std::vector<Point3D> points_;
    int max_iter_ = 30;
    double convergence_threshold_ = 1e-15;

    Eigen::Matrix<double, 3, 3> skew_symmetric(
        const Eigen::Vector3d& v
    );
    Eigen::Matrix<double, 4, 6> get_dproj_dxi(
        const Eigen::Vector3d& point
    );
    std::tuple<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>> compute_H_b
    (
        Eigen::Vector3d point3d,
        Eigen::Vector2d point2d,  
        Eigen::Vector2d projected_2d,
        Eigen::Matrix3d R_ini,
        Eigen::Vector3d t_ini,
        Eigen::Matrix<double, 2, 4> jacobian
    );
};

