// Stubs for common_ops_sm90/sm100 on ppc64le/V100: SM100 MXFP8 kernels are not compiled.
#include <torch/all.h>

void es_sm100_mxfp8_blockscaled_grouped_mm(
    const torch::Tensor& /*a*/,
    const torch::Tensor& /*b*/,
    const torch::Tensor& /*sfa*/,
    const torch::Tensor& /*sfb*/,
    torch::Tensor& /*d*/,
    const torch::Tensor& /*problem_sizes*/,
    const torch::Tensor& /*expert_offsets*/,
    const torch::Tensor& /*blockscale_offsets*/) {
  TORCH_CHECK(
      false,
      "es_sm100_mxfp8_blockscaled_grouped_mm is not available in this sm70/ppc64le kernel build.");
}

void es_sm100_mxfp8_blockscaled_grouped_quant(
    const torch::Tensor& /*input*/,
    const torch::Tensor& /*problem_sizes*/,
    const torch::Tensor& /*expert_offsets*/,
    const torch::Tensor& /*blockscale_offsets*/,
    torch::Tensor& /*quant_output*/,
    torch::Tensor& /*scale_factor*/) {
  TORCH_CHECK(
      false,
      "es_sm100_mxfp8_blockscaled_grouped_quant is not available in this sm70/ppc64le kernel build.");
}
