#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> bi_dd_wkv_forward_cuda(
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor k,
    torch::Tensor v
);

std::vector<torch::Tensor> bi_dd_wkv_backward_cuda(
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gy,
    torch::Tensor Af,
    torch::Tensor Bf,
    torch::Tensor Ab,
    torch::Tensor Bb,
    torch::Tensor y
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bi_dd_wkv_forward",  &bi_dd_wkv_forward_cuda,  "Bi-DD-WKV forward (CUDA)");
    m.def("bi_dd_wkv_backward", &bi_dd_wkv_backward_cuda, "Bi-DD-WKV backward (CUDA)");
}

