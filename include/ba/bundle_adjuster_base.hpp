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

class BundleAdjusterBase 
{
public:
    BundleAdjusterBase() = default;
    ~BundleAdjusterBase() = default;

    virtual std::vector<OptResult> optimize() = 0;

    std::vector<CameraModelBase> get_cameras()
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

protected:
    std::vector<CameraModelBase> cameras_;
    std::vector<Observation> observations_;
    std::vector<Point3D> points_;
};
