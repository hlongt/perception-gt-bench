#ifndef GTBENCH_H
#define GTBENCH_H

#if defined(_WIN32) || defined(__MINGW32__) || defined(__CYGWIN__)
#define GTB_API __declspec(dllexport)
#else
#define GTB_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

GTB_API void gtb_lrc_filter(const float *dl, const float *dr_gathered,
                            unsigned char *valid, int n, float thresh);

GTB_API void gtb_confidence(const float *c0, const float *c1, float *out, int n);

GTB_API void gtb_median3(const float *in, float *out, int h, int w);

GTB_API double gtb_absrel(const float *pred, const float *gt,
                          const unsigned char *valid, int n);

GTB_API void gtb_delta_counts(const float *pred, const float *gt,
                              const unsigned char *valid, int n, double thr,
                              long long *hit, long long *total);

#ifdef __cplusplus
}
#endif
#endif
