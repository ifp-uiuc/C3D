#ifndef UNPOOL3D_LAYER_HPP_
#define UNPOOL3D_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Unpooling3DLayer : public Layer<Dtype> {
 public:
  explicit Unpooling3DLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  int kernel_size_;
  int kernel_depth_;
  int stride_;
  int temporal_stride_;
  int pad_;
  int channels_;
  int length_;
  int height_;
  int width_;
  int unpooled_length_;
  int unpooled_height_;
  int unpooled_width_;
  Blob<Dtype> rand_idx_;

};

}

#endif /* UNPOOL3D_LAYER_HPP_ */