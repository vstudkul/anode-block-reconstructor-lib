#pragma once
#include <pcl/registration/warp_point_rigid.h>

// callback for the 2D Iterative Closest Point algorithm
template <typename PointSourceT, typename PointTargetT, typename Scalar = float>
class WarpPointRigid3DTrans
    : public pcl::registration::WarpPointRigid<PointSourceT, PointTargetT,
                                               Scalar> {
public:
  using Matrix4 =
      typename pcl::registration::WarpPointRigid<PointSourceT, PointTargetT,
                                                 Scalar>::Matrix4;
  using VectorX =
      typename pcl::registration::WarpPointRigid<PointSourceT, PointTargetT,
                                                 Scalar>::VectorX;

  using Ptr = pcl::shared_ptr<WarpPointRigid3DTrans<PointSourceT, PointTargetT, Scalar>>;
  using ConstPtr = pcl::shared_ptr<const WarpPointRigid3DTrans<PointSourceT, PointTargetT, Scalar>>;

  /** \brief Constructor. */
  WarpPointRigid3DTrans()
      : pcl::registration::WarpPointRigid<PointSourceT, PointTargetT, Scalar>(
            3) {}

  /** \brief Empty destructor */
  ~WarpPointRigid3DTrans() {}

  /** \brief Set warp parameters.
   * \param[in] p warp parameters (tx ty rz)
   */
  void setParam(const VectorX &p) override {
    assert(p.rows() == this->getDimension());

    // Copy the rotation and translation components
    this->transform_matrix_.setZero();
    this->transform_matrix_.setIdentity();
    this->transform_matrix_(0, 3) = p[0];
    this->transform_matrix_(1, 3) = p[1];
    this->transform_matrix_(2, 3) = 0; // no Z
  }
};
