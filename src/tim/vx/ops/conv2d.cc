/****************************************************************************
*
*    Copyright (c) 2020-2023 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
#include "tim/vx/ops/conv2d.h"

#include "builtin_op_impl.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

Conv2d::Conv2d(Graph* graph, PadType padding,
               const std::array<uint32_t, 2>& stride,
               const std::array<uint32_t, 2>& dilation, int32_t multiplier,
               DataLayout input_layout, DataLayout kernel_layout)
    : Conv2d(graph, 0, padding, {0, 0}, stride, dilation, {0, 0, 0, 0},
             multiplier, input_layout, kernel_layout) {}

Conv2d::Conv2d(Graph* graph, const std::array<uint32_t, 4> pad,
               const std::array<uint32_t, 2>& stride,
               const std::array<uint32_t, 2>& dilation, int32_t multiplier,
               DataLayout input_layout, DataLayout kernel_layout)
    : Conv2d(graph, 0, PadType::AUTO, {0, 0}, stride, dilation, pad,
             multiplier, input_layout, kernel_layout) {}

Conv2d::Conv2d(Graph* graph, int32_t weights, PadType padding,
               const std::array<uint32_t, 2>& ksize,
               const std::array<uint32_t, 2>& stride,
               const std::array<uint32_t, 2>& dilation, int32_t multiplier,
               DataLayout input_layout, DataLayout kernel_layout)
    : Conv2d(graph, weights, padding, ksize, stride, dilation, {0, 0, 0, 0},
             multiplier, input_layout, kernel_layout) {}

Conv2d::Conv2d(Graph* graph, int32_t weights, PadType padding,
               const std::array<uint32_t, 2>& ksize,
               const std::array<uint32_t, 2>& stride,
               const std::array<uint32_t, 2>& dilation,
               const std::array<uint32_t, 4>& pad, int32_t multiplier,
               DataLayout input_layout, DataLayout kernel_layout)
    : BuiltinOp(graph, VSI_NN_OP_CONV2D, 0, 0, input_layout),
      weights_(weights),
      padding_(padding),
      ksize_(ksize),
      stride_(stride),
      dilation_(dilation),
      pad_(pad),
      multiplier_(multiplier),
      kernel_layout_(kernel_layout) {
  this->impl()->node()->nn_param.conv2d.stride[0] = stride_[0];
  this->impl()->node()->nn_param.conv2d.stride[1] = stride_[1];
  this->impl()->node()->nn_param.conv2d.pad_type = TranslatePadType(padding_);
  this->impl()->node()->nn_param.conv2d.group = 1;
  this->impl()->node()->nn_param.conv2d.dilation[0] = dilation_[0];
  this->impl()->node()->nn_param.conv2d.dilation[1] = dilation_[1];
  this->impl()->node()->nn_param.conv2d.pad[0] = pad_[0];
  this->impl()->node()->nn_param.conv2d.pad[1] = pad_[1];
  this->impl()->node()->nn_param.conv2d.pad[2] = pad_[2];
  this->impl()->node()->nn_param.conv2d.pad[3] = pad_[3];
  this->impl()->node()->nn_param.conv2d.multiplier = multiplier_;
}

std::shared_ptr<Operation> Conv2d::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Conv2d>(
      this->weights_, this->padding_, this->ksize_, this->stride_,
      this->dilation_, this->pad_, this->multiplier_, this->impl_->layout_,
      this->kernel_layout_);
}

const std::vector<std::shared_ptr<Tensor>> Conv2d::ConstantInputsTensor() const {
   if (this->IsAllInputsConst()) {
    return {this->impl_->inputs_tensor_[0]};
  } else {
    return {};
  }
}

// Handle float16 bias if clang compiler is no less than 15.0.0 version
#ifdef TIM_VX_OPS_CONV2D_WITH_F16BIAS
void Conv2d::OnBindInputPostProc(const std::shared_ptr<Tensor>& tensor,
                                 int32_t input_idx) {
  if (tensor->GetDataType() == vx::DataType::FLOAT16 &&
      tensor->IsConstTensor() && impl_->inputs_tensor_.size() == 3) {
    uint32_t bias_size = 1;
    for (auto i : tensor->GetShape()) {
      bias_size *= i;
    }
    std::vector<_Float16> in(bias_size);
    tensor->CopyDataFromTensor(in.data());

    std::vector<float> out(bias_size);
    for (uint i = 0; i < bias_size; i++) {
      out[i] = static_cast<float>(in[i]);
    }
    TensorSpec fp32bias_spec(tim::vx::DataType::FLOAT32, tensor->GetShape(),
                             tim::vx::TensorAttribute::CONSTANT);
    auto out_tensor = impl_->graph_->CreateTensor(fp32bias_spec, out.data());

    impl_->inputs_tensor_[2] = out_tensor;
    impl_->node()->input.tensors[input_idx] = out_tensor->GetId();
    impl_->graph_->RenewTensorConsumersMap(tensor, out_tensor, this);
  }
}
#endif

}  // namespace ops
}  // namespace vx
}  // namespace tim