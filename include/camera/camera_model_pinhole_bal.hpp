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
        ) : f_(f), k1_(k1), k2_(k2), rot_(rot), t_(t), camera_idx_(camera_idx)
        {
            set_camera_model_name(CameraModelName::PINHOLE);
        }
        ~CameraModelPinholeBal(){}
        
        //getter & setter
        double get_fx() const { return f_; }
        double get_k1() const { return k1_; }
        double get_k2() const { return k2_; }
        Eigen::Vector3d get_rotation() const { return rot_; }
        Eigen::Vector3d get_translation() const { return t_; }
        void set_camera_idx(int idx) { camera_idx_ = idx; }

        // functions
        Eigen::Matrix<double, 2, 4> get_prj_jacobian(const Eigen::Vector3d &X) const;
        Eigen::Vector2d project(
            const Eigen::Vector3d &xyz,
            const Eigen::Matrix3d &R,
            const Eigen::Vector3d &t
        ) const;
        Eigen::Matrix3d rotationMatrix() const;
        // // Implement pure virtual functions from CameraModelBase
        // cv::Mat image_to_normalcoord3d(const std::vector<cv::Point2f> &pixels) const override;
        // cv::Point2f normalcoord3d_to_image(const cv::Point3f &points_normalized) const override;
        // bool set_intrinsic(std::string data_file_path) override;
        
    private:    
        // parameters
        Eigen::Vector3d rot_; // Rodrigues
        Eigen::Vector3d t_;
        double f_, k1_, k2_;
        int camera_idx_;
};
