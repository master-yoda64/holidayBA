#pragma once
#include <bits/stdc++.h>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <camera/camera_model_pinhole_bal.hpp>
#include <camera/camera_type_traits.hpp>

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

template<typename CameraModelT>
class BundleAdjusterBase 
{
public:
    BundleAdjusterBase() = default;
    ~BundleAdjusterBase() = default;

    virtual std::vector<OptResult> optimize() = 0;
    virtual OptResult optimize_camera(
        const std::vector<Observation>& obs,
        const CameraModelType_t<CameraModelT>& cam
    ) = 0;
    std::vector<Observation> get_observations()
    {
        return observations_;
    }
    std::vector<Point3D> get_points()
    {
        return points_;
    }
    std::vector<CameraModelType_t<CameraModelT>> get_cameras()
    {
        return cameras_;
    }

protected:
    std::vector<Observation> observations_;
    std::vector<Point3D> points_;
    std::vector<CameraModelType_t<CameraModelT>> cameras_;
};