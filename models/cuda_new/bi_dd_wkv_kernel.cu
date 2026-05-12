#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define EPS 1e-6

// 为 float 单独用 __expf，加速一点
template <typename T>
__device__ __forceinline__ T my_exp(T x) { return exp(x); }

template <>
__device__ __forceinline__ float my_exp<float>(float x) { return __expf(x); }

// ==================
// Forward Kernel
// ==================
template <typename scalar_t>
__global__ void bi_dd_wkv_forward_kernel(
    int B, int T, int C,
    const scalar_t* __restrict__ w,   // [B,T,C]
    const scalar_t* __restrict__ u,   // [C]
    const scalar_t* __restrict__ k,   // [B,T,C]
    const scalar_t* __restrict__ v,   // [B,T,C]
    scalar_t* __restrict__ y,         // [B,T,C]
    scalar_t* __restrict__ Af,        // [B,T,C]
    scalar_t* __restrict__ Bf,        // [B,T,C]
    scalar_t* __restrict__ Ab,        // [B,T,C]
    scalar_t* __restrict__ Bb         // [B,T,C]
) {
    int bc = blockIdx.x * blockDim.x + threadIdx.x;
    if (bc >= B * C) return;

    int b = bc / C;
    int c = bc % C;

    // ---------- 1. Forward branch: 只积累过去 ----------
    scalar_t a_f = 0;
    scalar_t b_f = 0;

    for (int t = 0; t < T; ++t) {
        int idx = (b * T + t) * C + c;

        // Af/Bf 存的是「到 t 为止，但不含 t」的状态
        Af[idx] = a_f;
        Bf[idx] = b_f;

        scalar_t wt = w[idx];
        scalar_t kt = k[idx];
        scalar_t vt = v[idx];

        scalar_t ew = my_exp<scalar_t>(-wt);
        scalar_t ek = my_exp<scalar_t>( kt);

        a_f = ew * a_f + ek * vt;
        b_f = ew * b_f + ek;
    }

    // ---------- 2. Backward branch: 只积累未来 + 直接算 y ----------
    scalar_t a_b = 0;
    scalar_t b_b = 0;
    scalar_t u_c = u[c];

    for (int t = T - 1; t >= 0; --t) {
        int idx = (b * T + t) * C + c;

        // Ab/Bb 存的是「未来但不含 t」的状态
        Ab[idx] = a_b;
        Bb[idx] = b_b;

        // 计算输出 y_t
        scalar_t Af_t = Af[idx];
        scalar_t Bf_t = Bf[idx];
        scalar_t kt   = k[idx];
        scalar_t vt   = v[idx];

        scalar_t euk = my_exp<scalar_t>(u_c + kt);

        scalar_t num = Af_t + a_b + euk * vt;
        scalar_t den = Bf_t + b_b + euk + (scalar_t)EPS;

        y[idx] = num / den;

        // 更新未来状态到更早的 t-1
        scalar_t wt = w[idx];
        scalar_t ew = my_exp<scalar_t>(-wt);
        scalar_t ek = my_exp<scalar_t>( kt);

        a_b = ew * a_b + ek * vt;
        b_b = ew * b_b + ek;
    }
}

// ==================
// Backward Kernel
// ==================
template <typename scalar_t>
__global__ void bi_dd_wkv_backward_kernel(
    int B, int T, int C,
    const scalar_t* __restrict__ w,    // [B,T,C]
    const scalar_t* __restrict__ u,    // [C]
    const scalar_t* __restrict__ k,    // [B,T,C]
    const scalar_t* __restrict__ v,    // [B,T,C]
    const scalar_t* __restrict__ gy,   // [B,T,C]

    const scalar_t* __restrict__ Af,   // [B,T,C]
    const scalar_t* __restrict__ Bf,   // [B,T,C]
    const scalar_t* __restrict__ Ab,   // [B,T,C]
    const scalar_t* __restrict__ Bb,   // [B,T,C]
    const scalar_t* __restrict__ y,    // [B,T,C]

    scalar_t* __restrict__ gw,         // [B,T,C]
    scalar_t* __restrict__ gu,         // [C]
    scalar_t* __restrict__ gk,         // [B,T,C]
    scalar_t* __restrict__ gv,         // [B,T,C]

    scalar_t* __restrict__ gAf,        // [B,T,C]
    scalar_t* __restrict__ gBf,        // [B,T,C]
    scalar_t* __restrict__ gAb,        // [B,T,C]
    scalar_t* __restrict__ gBb         // [B,T,C]
) {
    int bc = blockIdx.x * blockDim.x + threadIdx.x;
    if (bc >= B * C) return;

    int b = bc / C;
    int c = bc % C;

    scalar_t gu_c = 0;

    // ---------- Pass 1: 从 y,gy 分解到 Af,Bf,Ab,Bb,u,k,v ----------
    for (int t = 0; t < T; ++t) {
        int idx = (b * T + t) * C + c;

        scalar_t Af_t = Af[idx];
        scalar_t Bf_t = Bf[idx];
        scalar_t Ab_t = Ab[idx];
        scalar_t Bb_t = Bb[idx];
        scalar_t kt   = k[idx];
        scalar_t vt   = v[idx];
        scalar_t yt   = y[idx];
        scalar_t gy_t = gy[idx];
        scalar_t u_c  = u[c];

        scalar_t euk = my_exp<scalar_t>(u_c + kt);
        scalar_t num = Af_t + Ab_t + euk * vt;
        scalar_t den = Bf_t + Bb_t + euk + (scalar_t)EPS;
        scalar_t z   = (scalar_t)1.0 / den;

        // y = num / den
        scalar_t g_num = gy_t * z;
        scalar_t g_den = -gy_t * yt * z;

        scalar_t gAf_t = g_num;
        scalar_t gAb_t = g_num;
        scalar_t gBf_t = g_den;
        scalar_t gBb_t = g_den;

        // num 中的 euk * v
        scalar_t g_p   = g_num;
        scalar_t g_euk = g_den + g_p * vt;
        scalar_t g_v   = g_p * euk;

        // euk = exp(u + k)
        scalar_t g_s   = g_euk * euk;  // s = u + k
        gu_c += g_s;
        scalar_t g_k_euk = g_s;

        gAf[idx] = gAf_t;
        gBf[idx] = gBf_t;
        gAb[idx] = gAb_t;
        gBb[idx] = gBb_t;

        gv[idx] += g_v;
        gk[idx] += g_k_euk;
    }

    // ---------- Pass 2: forward branch (Af,Bf) 递推反传 ----------
    scalar_t grad_a_next = 0;
    scalar_t grad_b_next = 0;

    for (int t = T - 1; t >= 0; --t) {
        int idx = (b * T + t) * C + c;

        scalar_t a_t   = Af[idx];
        scalar_t b_t   = Bf[idx];
        scalar_t gA_t  = gAf[idx];
        scalar_t gB_t  = gBf[idx];
        scalar_t wt    = w[idx];
        scalar_t kt    = k[idx];
        scalar_t vt    = v[idx];

        scalar_t ew = my_exp<scalar_t>(-wt);
        scalar_t ek = my_exp<scalar_t>( kt);

        // 递推链式求导
        scalar_t g_a_t = gA_t + grad_a_next * ew;
        scalar_t g_b_t = gB_t + grad_b_next * ew;

        scalar_t g_ew = grad_a_next * a_t + grad_b_next * b_t;
        scalar_t g_ek = grad_a_next * vt  + grad_b_next * (scalar_t)1.0;

        // ew = exp(-w) => d ew / d w = -ew
        gw[idx] += g_ew * (-ew);

        // ek = exp(k) => d ek / d k = ek
        gk[idx] += g_ek * ek;

        // a_next = ew * a_t + ek * v_t
        gv[idx] += grad_a_next * ek;

        grad_a_next = g_a_t;
        grad_b_next = g_b_t;
    }

    // ---------- Pass 3: backward branch (Ab,Bb) 递推反传 ----------
    grad_a_next = 0;
    grad_b_next = 0;

    for (int r = T - 1; r >= 0; --r) {
        int t   = T - 1 - r;
        int idx = (b * T + t) * C + c;

        scalar_t a_t   = Ab[idx];
        scalar_t b_t   = Bb[idx];
        scalar_t gA_t  = gAb[idx];
        scalar_t gB_t  = gBb[idx];
        scalar_t wt    = w[idx];
        scalar_t kt    = k[idx];
        scalar_t vt    = v[idx];

        scalar_t ew = my_exp<scalar_t>(-wt);
        scalar_t ek = my_exp<scalar_t>( kt);

        scalar_t g_a_t = gA_t + grad_a_next * ew;
        scalar_t g_b_t = gB_t + grad_b_next * ew;

        scalar_t g_ew = grad_a_next * a_t + grad_b_next * b_t;
        scalar_t g_ek = grad_a_next * vt  + grad_b_next * (scalar_t)1.0;

        gw[idx] += g_ew * (-ew);
        gk[idx] += g_ek * ek;
        gv[idx] += grad_a_next * ek;

        grad_a_next = g_a_t;
        grad_b_next = g_b_t;
    }

    // ---------- 写回 gu[c] ----------
    if (gu_c != (scalar_t)0) {
        atomicAdd(&gu[c], gu_c);
    }
}

// ==================
// C++ 封装（给 .cpp 调用）
// ==================

std::vector<torch::Tensor> bi_dd_wkv_forward_cuda(
    torch::Tensor w,   // [B,T,C]
    torch::Tensor u,   // [C]
    torch::Tensor k,   // [B,T,C]
    torch::Tensor v    // [B,T,C]
) {
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(u.is_cuda(), "u must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA");

    auto B = k.size(0);
    auto T = k.size(1);
    auto C = k.size(2);

    TORCH_CHECK(w.sizes() == k.sizes(), "w and k must have same shape");
    TORCH_CHECK(v.sizes() == k.sizes(), "v and k must have same shape");
    TORCH_CHECK(u.size(0) == C, "u.shape must be [C]");

    auto y  = torch::zeros_like(k);
    auto Af = torch::empty_like(k);
    auto Bf = torch::empty_like(k);
    auto Ab = torch::empty_like(k);
    auto Bb = torch::empty_like(k);

    int threads = 256;
    int blocks  = (B * C + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(k.scalar_type(), "bi_dd_wkv_forward_cuda", [&] {
        bi_dd_wkv_forward_kernel<scalar_t><<<blocks, threads>>>(
            B, T, C,
            w.data_ptr<scalar_t>(),
            u.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            Af.data_ptr<scalar_t>(),
            Bf.data_ptr<scalar_t>(),
            Ab.data_ptr<scalar_t>(),
            Bb.data_ptr<scalar_t>()
        );
    });

    return {y, Af, Bf, Ab, Bb};
}

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
) {
    auto B = k.size(0);
    auto T = k.size(1);
    auto C = k.size(2);

    auto gw  = torch::zeros_like(w);
    auto gu  = torch::zeros_like(u);
    auto gk  = torch::zeros_like(k);
    auto gv  = torch::zeros_like(v);

    auto gAf = torch::empty_like(Af);
    auto gBf = torch::empty_like(Bf);
    auto gAb = torch::empty_like(Ab);
    auto gBb = torch::empty_like(Bb);

    int threads = 256;
    int blocks  = (B * C + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(k.scalar_type(), "bi_dd_wkv_backward_cuda", [&] {
        bi_dd_wkv_backward_kernel<scalar_t><<<blocks, threads>>>(
            B, T, C,
            w.data_ptr<scalar_t>(),
            u.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            gy.data_ptr<scalar_t>(),

            Af.data_ptr<scalar_t>(),
            Bf.data_ptr<scalar_t>(),
            Ab.data_ptr<scalar_t>(),
            Bb.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),

            gw.data_ptr<scalar_t>(),
            gu.data_ptr<scalar_t>(),
            gk.data_ptr<scalar_t>(),
            gv.data_ptr<scalar_t>(),

            gAf.data_ptr<scalar_t>(),
            gBf.data_ptr<scalar_t>(),
            gAb.data_ptr<scalar_t>(),
            gBb.data_ptr<scalar_t>()
        );
    });

    return {gw, gu, gk, gv};
}

