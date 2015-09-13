#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
SegmentationDataLayer<Dtype>::~SegmentationDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void SegmentationDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  SegmentationDataParameter data_param = this->layer_param_.segmentation_data_param();
  const string source = data_param.source();
  const bool shuffle = data_param.shuffle();
  const int batch_size = data_param.batch_size();

  for(int i = 0; i < data_param.mean_value_size(); ++i){
    mean_values_.push_back(data_param.mean_value(i));
  }

  string image_path;
  string gt_image_path;

  LOG(INFO) << "Opening file " << source;
  std::ifstream in_file(source.c_str());
  while(in_file >> image_path >> gt_image_path){
    std::pair<string, string> pair = std::make_pair(image_path, gt_image_path);
    image_pairs_.push_back(pair);
  }

  LOG(INFO) << "Total number of image pairs: " << image_pairs_.size();

  image_pair_id_ = 0;
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));

  if(shuffle){
    ShuffleImages();
  }

  cv::Mat I = cv::imread(image_pairs_[0].first);
  int height = I.rows;
  int width = I.cols;
  const int crop_size = this->layer_param_.transform_param().crop_size();
  if(crop_size > 0){
      height = crop_size;
      width = crop_size;
  }
  LOG(INFO) << height << "," << width;

  this->transformed_data_.Reshape(1, 4, height, width);

  int image_shape_array[4] = {batch_size, 4, height, width};
  vector<int> top_shape(&image_shape_array[0], &image_shape_array[0] + 4);
  this->prefetch_data_.Reshape(top_shape);
  top[0]->Reshape(top_shape);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
}

template <typename Dtype>
void SegmentationDataLayer<Dtype>::ShuffleImages() {
    LOG(INFO) << "Shuffle images";
    caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(image_pairs_.begin(), image_pairs_.end(), prefetch_rng);
}

template <typename Dtype>
void SegmentationDataLayer<Dtype>::InternalThreadEntry() {

  SegmentationDataParameter data_param = this->layer_param_.segmentation_data_param();
  const int batch_size = data_param.batch_size();
  const bool shuffle = data_param.shuffle();

  cv::Mat I = cv::imread(image_pairs_[image_pair_id_].first);
  int input_height = I.rows;
  int input_width = I.cols;
  int height = I.rows;
  int width = I.cols;
  const int crop_size = this->layer_param_.transform_param().crop_size();
  if(crop_size > 0){
      height = crop_size;
      width = crop_size;
  }

  this->transformed_data_.Reshape(1, 4, height, width);

  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();

  Datum datum;
  datum.set_channels(4);
  datum.set_height(input_height);
  datum.set_width(input_width);

  for(int item_id = 0; item_id < batch_size; item_id++){
    datum.clear_data();
    datum.clear_float_data();

    cv::Mat I_image = cv::imread(image_pairs_[image_pair_id_].first);
    for(int c = 0; c < I.channels(); ++c){
      for(int h = 0; h < I_image.rows; ++h){
        for(int w = 0; w < I_image.cols; ++w){
            datum.add_float_data((float)I_image.at<cv::Vec3b>(h,w)[c] - mean_values_[c]);
        }
      }
    }

    cv::Mat I_label = cv::imread(image_pairs_[image_pair_id_].second, CV_LOAD_IMAGE_GRAYSCALE);
    for(int h = 0; h < I_label.rows; ++h){
        for(int w = 0; w < I_label.cols; ++w){
            float label = I_label.at<uchar>(h, w) > 0 ? 1 : 0;
            datum.add_float_data(label);
        }
    }

    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));

    image_pair_id_++;
    if(image_pair_id_ >= image_pairs_.size()){
      image_pair_id_ = 0;
      if(shuffle){
        ShuffleImages();
      }
    }

  }
}

INSTANTIATE_CLASS(SegmentationDataLayer);
REGISTER_LAYER_CLASS(SegmentationData);

}  // namespace caffe


