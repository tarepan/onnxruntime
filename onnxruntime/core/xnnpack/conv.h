// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

struct xnn_operator;

namespace onnxruntime {
namespace xnnpack {

class Convolution2d : public OpKernel {
 public:
  Convolution2d(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  struct xnn_operator* op0 = nullptr;
  TensorShape output_shape;
};

class DepthWiseConvolution2d : public OpKernel {
 public:
  DepthWiseConvolution2d(const OpKernelInfo& info);
  Status Compute(OpKernelContext*) const override;

 private:
  struct xnn_operator* op0 = nullptr;
  TensorShape output_shape;
  float* weight_ = nullptr;
};
}  // namespace xnnpack
}  // namespace onnxruntime
