#include "gtbench.h"

#include <cmath>
#include <cstddef>

extern "C" {

GTB_API double gtb_absrel(const float *pred, const float *gt,
                          const unsigned char *valid, int n) {
    double sum = 0.0;
    long long count = 0;
    for (int i = 0; i < n; ++i) {
        if (!valid[i]) continue;
        const double g = gt[i];
        if (g < 1e-6) continue;
        double p = pred[i];
        if (p < 1e-6) p = 1e-6;
        sum += std::fabs(p - g) / g;
        ++count;
    }
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

GTB_API void gtb_delta_counts(const float *pred, const float *gt,
                              const unsigned char *valid, int n, double thr,
                              long long *hit, long long *total) {
    long long h = 0, t = 0;
    for (int i = 0; i < n; ++i) {
        if (!valid[i]) continue;
        const double g = gt[i];
        if (g < 1e-6) continue;
        double p = pred[i];
        if (p < 1e-6) p = 1e-6;
        const double ratio = p / g > g / p ? p / g : g / p;
        if (ratio < thr) ++h;
        ++t;
    }
    *hit = h;
    *total = t;
}

}
