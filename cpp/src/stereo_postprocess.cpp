#include "gtbench.h"

#include <cmath>
#include <cstddef>

extern "C" {

GTB_API void gtb_lrc_filter(const float *dl, const float *dr_gathered,
                            unsigned char *valid, int n, float thresh) {
    for (int i = 0; i < n; ++i) {
        const float d = std::fabs(dl[i] - dr_gathered[i]);
        valid[i] = (d <= thresh && dl[i] > 0.5f) ? 1u : 0u;
    }
}

GTB_API void gtb_confidence(const float *c0, const float *c1, float *out, int n) {
    for (int i = 0; i < n; ++i) {
        const float denom = c1[i] > 1e-6f ? c1[i] : 1e-6f;
        float v = 1.0f - c0[i] / denom;
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        out[i] = v;
    }
}

GTB_API void gtb_median3(const float *in, float *out, int h, int w) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float a = in[y * w + x];
            float b = in[y * w + (x > 0 ? x - 1 : x)];
            float c = in[y * w + (x + 1 < w ? x + 1 : x)];
            float lo = a < b ? a : b;
            float hi = a > b ? a : b;
            if (c < lo) { out[y * w + x] = lo; }
            else if (c > hi) { out[y * w + x] = hi; }
            else { out[y * w + x] = c; }
        }
    }
}

}
