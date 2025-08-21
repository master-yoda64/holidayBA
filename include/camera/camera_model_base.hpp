#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

enum CameraModelName
{
    NONE=0,
    PINHOLE = 1,
    EQUIRECTANGULAR = 2,
    NameCount
};

class CameraModelBase
{
    public:
        CameraModelBase() = default;
        ~CameraModelBase() = default;

        //getter & setter
        virtual int get_camera_id() const = 0;
        virtual CameraModelName get_camera_model_name() const = 0;
        virtual void set_camera_model_name(CameraModelName name) = 0;
        virtual void set_camera_id(int id) = 0;

        // functions
        /**
         * @brief Get normalized image coordinates from pixel coordinates
         * @param pixel Pixel coordinates
         * @return 2d coordinates in normalized image plane
         */
        // virtual cv::Mat image_to_normalcoord3d(const std::vector<cv::Point2f> &pixels) const = 0;

        // virtual cv::Point2f normalcoord3d_to_image(const cv::Point3f &points_normalized) const = 0;

        // // cv::Point2f coord3d_to_image(const cv::Point3f &points_3d)
        // // {
        // //     cv::Point3f points_normalized = xyz_to_homo(points_3d);
        // //     return this->normalcoord3d_to_image(points_normalized);
        // // };
        // virtual bool set_intrinsic(std::string data_file_path) = 0; // set parameters for each camera model
    protected:    
        // parameters
        int camera_idx_;
        CameraModelName camera_model_name_ = CameraModelName::NONE;
        //cv::Mat K = cv::Mat::empty(3, 3, CV_32F); // camera matrix
};
