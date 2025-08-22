#pragma once

#include <Eigen/Core>
#include "camera_model_base.hpp"

class CameraModelPinholeBal : public CameraModelBase
{

    public:
        CameraModelPinholeBal(
            double f, 
            double k1, 
            double k2, 
            const Eigen::Vector3d& rot, 
            const Eigen::Vector3d& t,
            int camera_idx
        ) : f_(f), k1_(k1), k2_(k2), rot_(rot), t_(t)
        {
            set_camera_model_name(CameraModelName::PINHOLE);
            camera_idx_ = camera_idx;
        }
        ~CameraModelPinholeBal(){}
        
        //getter & setter
        double get_fx() const { return f_; }
        double get_k1() const { return k1_; }
        double get_k2() const { return k2_; }
        Eigen::Vector3d get_rotation() const { return rot_; }
        Eigen::Vector3d get_translation() const { return t_; }

        // functions
        Eigen::Matrix<double, 2, 4> get_prj_jacobian(const Eigen::Vector3d &X) const;
        Eigen::Vector2d project(
            const Eigen::Vector3d &xyz,
            const Eigen::Matrix3d &R,
            const Eigen::Vector3d &t
        ) const;
        Eigen::Matrix3d rotation_matrix() const;
    private:    
        // parameters
        Eigen::Vector3d rot_; // Rodrigues
        Eigen::Vector3d t_;
        double f_, k1_, k2_;
};
