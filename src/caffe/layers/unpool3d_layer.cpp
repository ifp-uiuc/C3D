#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/unpooling3d_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void Unpooling3DLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "UnpoolingLayer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "UnpoolingLayer takes a single blob as output.";
  kernel_size_ = this->layer_param_.unpooling_param().kernel_size();
  kernel_depth_ = this->layer_param_.unpooling_param().kernel_depth();
  stride_ = this->layer_param_.unpooling_param().stride();
  temporal_stride_ = this->layer_param_.unpooling_param().temporal_stride();
  pad_ = this->layer_param_.unpooling_param().pad();
  if (pad_ != 0) {
    CHECK_EQ(this->layer_param_.unpooling_param().unpool(),
             UnpoolingParameter_UnpoolMethod_MAX)
        << "Padding implemented only for MAX unpooling.";
  }
  channels_ = bottom[0]->channels();
  length_ = bottom[0]->length();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  //not sure? use DeconvNet or SegNet
  //DeconvNet: unpooled_height_ = max((height_ - 1) * stride_h_ + kernel_h_ - 2 * pad_h_, height_ * stride_h_ - pad_h_ + 1);
  //SegNet: unpooled_height_= height_ * stride_ - pad_;
  unpooled_height_= height_ * stride_ - pad_ ;
  unpooled_width_ = width_ * stride_ - pad_;
  unpooled_length_ = length_ * stride_ - pad_;
  (*top)[0]->Reshape(bottom[0]->num(), channels_, unpooled_length_, unpooled_height_,
      unpooled_width_);
}

template <typename Dtype>
void Unpooling3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_mask_data =bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  //initialize
  const int top_count = top[0]->count();
  caffe_set(top_count,Dtype(0), top_data);
  // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int i =0;i < length_ * height_ * width_; ++i){
          const int idx =static_cast<int>(bottom_mask_data[i]);
          if( idx >= unpooled_length_ * unpooled_height_ * unpooled_width_){
            LOG(FATAL) << "upsample top index " << idx << " out of range - ";
          }
          top_data[idx] =bottom_data[i];
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        bottom_mask_data += bottom[1]->offset(0,1);
        top_data += top[0]->offset(0, 1); 
      }
    }
}

template <typename Dtype>
void Unpooling3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_mask_data = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  
  const int bottom_count=bottom[0]->count();
  caffe_set(bottom_count, Dtype(0), bottom_diff);
  
  // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for(i=0; i < length_ * height_ * width_; ++i){
          const int idx = static_cast<int>(bottom_mask_data[i]);
          if(idx >= length_ * height_ * width_){
            LOG(FATAL) << "upsample top index " << idx << " out of range";
          }
          bottom_diff[i]=top_diff[idx];
        }
        // compute offset
        bottom_diff += bottom[0]->offset(0, 1);
        bottom_mask_data +=bottom[1]->offset(0,1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    
}

INSTANTIATE_CLASS(Unpooling3DLayer);
}