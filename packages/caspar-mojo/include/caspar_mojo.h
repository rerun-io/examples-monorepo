/* C ABI of the caspar-mojo bundle-adjustment solver (libcaspar_mojo.so).
 *
 * The same array contract as caspar_mojo/problem.py, in float32: the solver works in float32 on the GPU, as
 * CASPAR's generated f32 tree does, so the caller converts once while it packs and nothing converts again.
 * Every struct field is 8 bytes wide (int64_t, double or a pointer), so the layout has no padding and matches the
 * Mojo side field for field. Arrays are row-major and caller-owned; the call copies them to the GPU, solves, and
 * writes the accepted state into `CasparMojoState`. Poses are [qx, qy, qz, qw, tx, ty, tz]; quaternions come back
 * unnormalised (the caller normalises at write-back, as COLMAP's CASPAR adapter does).
 *
 * Load with dlopen; check caspar_mojo_abi_version() == CASPAR_MOJO_ABI_VERSION before any other call.
 */
#ifndef CASPAR_MOJO_H_
#define CASPAR_MOJO_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CASPAR_MOJO_ABI_VERSION 1

#define CASPAR_MOJO_MAX_INTRINSICS 12

/* camera_model values (caspar_mojo.problem.CameraModel). */
#define CASPAR_MOJO_PINHOLE 0
#define CASPAR_MOJO_SIMPLE_RADIAL 1
#define CASPAR_MOJO_OPENCV_FISHEYE 2
#define CASPAR_MOJO_FULL_OPENCV 3

/* Return codes. */
#define CASPAR_MOJO_OK 0
#define CASPAR_MOJO_ERROR 1       /* the solver raised; error_message has the text */
#define CASPAR_MOJO_UNSUPPORTED 2 /* camera model or refine flags not compiled into this library */

/* exit_reason values: CASPAR's ExitReason. */
#define CASPAR_MOJO_EXIT_MAX_ITERATIONS 0
#define CASPAR_MOJO_EXIT_CONVERGED_DIAG_EXIT 1
#define CASPAR_MOJO_EXIT_CONVERGED_SCORE_THRESHOLD 2

typedef struct CasparMojoProblem {
  int64_t camera_model; /* one model for every camera of the problem */
  int64_t num_frames;
  int64_t num_cameras;
  int64_t num_points;
  int64_t num_observations;
  const float* intrinsics;        /* [num_cameras][12] COLMAP camera.params order, zero-padded */
  const uint8_t* intrinsics_free; /* [num_cameras] focal lengths + extra params refined (never the principal point) */
  const float* sensor_from_rig;   /* [num_cameras][7] identity for a reference sensor */
  const uint8_t* sensor_free;     /* [num_cameras] */
  const float* rig_from_world;    /* [num_frames][7] */
  const uint8_t* pose_free;       /* [num_frames] */
  const float* points;            /* [num_points][3] */
  const uint8_t* point_free;      /* [num_points] */
  const float* pixels;            /* [num_observations][2] */
  const int32_t* observation_frame;
  const int32_t* observation_camera;
  const int32_t* observation_point;
} CasparMojoProblem;

/* CASPAR's SolverParams under its names; the reduced camera system is the only mode. */
typedef struct CasparMojoOptions {
  int64_t device_id;
  int64_t solver_iter_max;
  double diag_init;
  double diag_min;
  double diag_scaling_up;
  double diag_scaling_down;
  double diag_exit_value;
  double score_exit_value;
  double solver_rel_decrease_min;
  int64_t reduced_pcg_iter_max;
  double reduced_pcg_eta;
} CasparMojoOptions;

/* Output arrays, same shapes as the problem's; fixed nodes come back unchanged. */
typedef struct CasparMojoState {
  float* intrinsics;
  float* sensor_from_rig;
  float* rig_from_world;
  float* points;
} CasparMojoState;

typedef struct CasparMojoStats {
  double initial_cost; /* 1/2 sum of squared residuals at the start */
  double final_cost;   /* the same at the accepted state */
  int64_t iterations;  /* LM trials (CASPAR's iteration_count) */
  int64_t pcg_iterations;
  int64_t exit_reason; /* CASPAR_MOJO_EXIT_* */
  double solve_seconds; /* the LM loop on the GPU, uploads and downloads excluded */
  double total_seconds; /* the whole call */
  int64_t launches;
} CasparMojoStats;

int32_t caspar_mojo_abi_version(void);

/* 1 if this library has the solver for the model and refine flags, else 0. */
int32_t caspar_mojo_supports(int64_t camera_model, int64_t refine_intrinsics, int64_t refine_rig);

/* CASPAR_MOJO_OK, CASPAR_MOJO_ERROR or CASPAR_MOJO_UNSUPPORTED. Refine flags follow from the masks. On an error
 * the message is written NUL-terminated into the caller's error_message (Mojo has no global variables, so the
 * library cannot keep a last-error string of its own). */
int32_t caspar_mojo_solve(const CasparMojoProblem* problem,
                          const CasparMojoOptions* options,
                          const CasparMojoState* result,
                          CasparMojoStats* stats,
                          char* error_message,
                          int64_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif /* CASPAR_MOJO_H_ */
