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
#ifndef TIM_LAYOUT_INFER_ACTIVATION_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_ACTIVATION_LAYOUT_INFERENCE_H_

#include <algorithm>

#include "tim/vx/ops/activations.h"

#include "ops/op_layout_inference.h"
#include "permute_vector.h"
#include "builtin_op_impl.h"

namespace tim {
namespace transform {
template <typename OpType>
class ActivationLayoutInfer : public OpLayoutInfer {
 public:
  ActivationLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    // Transmit input pv to out pv directly for activation ops
    assert(op_->impl()->InputsTensor().size() == 1);
    auto i_src = op_->impl()->InputsTensor()[0];
    auto input_pv = context_->GetPermuteVector(i_src);
    auto activation = op_->Clone(context_->infer_graph_);
    auto out_infer = CreateOutputsTensor(input_pv);
    (*activation)
        .BindInput(context_->GetMapedTensor(i_src))
        .BindOutput(out_infer[0]);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], input_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

class PReluLayoutInfer : public OpLayoutInfer {
 public:
  PReluLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto src_input = op_->impl()->InputsTensor()[0];
    auto input_pv = context_->GetPermuteVector(src_input);
    auto src_slope = op_->impl()->InputsTensor()[1];
    std::shared_ptr<tim::vx::Tensor> infer_slope;
    if (input_pv->IsAligned()) {
      infer_slope = context_->infer_graph_->CreateTensor(
          src_slope->GetSpec(), src_slope->GetDataRef());
    } else {
      
      auto new_spec = src_slope->GetSpec();
      tim::vx::ShapeType s;
      s.push_back(*(new_spec.shape_.end()-1));
      new_spec.shape_ = s;
      infer_slope = context_->infer_graph_->CreateTensor(
          new_spec, src_slope->GetDataRef());
    }
    context_->UpdateTensorMap(src_slope, infer_slope);
    context_->SetPermuteVector(src_slope,
                               MakeShared(src_slope->GetShape().size()));

    auto prelu =
        context_->infer_graph_->CreateOperation<vx::ops::Prelu>(MapAxis(
            input_pv->AsStdVec(), op_->impl()->node()->nn_param.prelu.axis));
    auto out_infer = CreateOutputsTensor(input_pv);
    for (const auto& i_src : op_->impl()->InputsTensor()) {
      (*prelu).BindInput(context_->GetMapedTensor(i_src));
    }
    (*prelu).BindOutput(out_infer[0]);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], input_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

using ReluLayoutInfer = ActivationLayoutInfer<vx::ops::Relu>;
using Relu1LayoutInfer = ActivationLayoutInfer<vx::ops::Relu1>;
using Relu6LayoutInfer = ActivationLayoutInfer<vx::ops::Relu6>;
using LeakyReluLayoutInfer = ActivationLayoutInfer<vx::ops::LeakyRelu>;
using EluLayoutInfer = ActivationLayoutInfer<vx::ops::Elu>;
using SigmoidLayoutInfer = ActivationLayoutInfer<vx::ops::Sigmoid>;
using MishLayoutInfer = ActivationLayoutInfer<vx::ops::Mish>;
using HardSigmoidLayoutInfer = ActivationLayoutInfer<vx::ops::HardSigmoid>;
using SoftReluLayoutInfer = ActivationLayoutInfer<vx::ops::SoftRelu>;
using HardSwishLayoutInfer = ActivationLayoutInfer<vx::ops::HardSwish>;
using TanhLayoutInfer = ActivationLayoutInfer<vx::ops::Tanh>;

}  // namespace transform
}  // namespace tim

#endif