/* Links against libcaspar_mojo.so like COLMAP's C++ would: solves a small exact PINHOLE problem from perturbed
 * points and poses, and checks the unsupported-model path reports through the error buffer. Exit 0 = pass. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "caspar_mojo.h"

enum { NF = 6, NP = 64, NO = NF * NP };

int main(void) {
  float intr[12] = {500, 500, 320, 240}, sfr[7] = {0, 0, 0, 1, 0, 0, 0}, poses[NF * 7], pts[NP * 3], pix[NO * 2];
  uint8_t intr_free = 0, sfr_free = 0, pose_free[NF], pt_free[NP];
  int32_t of[NO], oc[NO], op[NO];
  srand(7);
  for (int j = 0; j < NP; ++j) {
    for (int k = 0; k < 3; ++k) pts[3 * j + k] = (float)rand() / RAND_MAX * 2 - 1 + (k == 2 ? 5 : 0);
    pt_free[j] = 1;
  }
  for (int i = 0; i < NF; ++i) {
    const float p[7] = {0, 0, 0, 1, 0.2f * i - 0.5f, 0.1f * (i % 2), 0};
    memcpy(poses + 7 * i, p, sizeof(p));
    pose_free[i] = i > 0;
    for (int j = 0; j < NP; ++j) {
      const int o = i * NP + j;
      const float* x = pts + 3 * j;
      pix[2 * o] = 500 * (x[0] + p[4]) / (x[2] + p[6]) + 320;
      pix[2 * o + 1] = 500 * (x[1] + p[5]) / (x[2] + p[6]) + 240;
      of[o] = i, oc[o] = 0, op[o] = j;
    }
  }
  for (int j = 0; j < NP * 3; ++j) pts[j] += 0.02f * ((float)rand() / RAND_MAX - 0.5f);
  for (int i = 1; i < NF; ++i) poses[7 * i + 4] += 0.01f;
  CasparMojoProblem pr = {CASPAR_MOJO_PINHOLE, NF, 1, NP, NO, intr, &intr_free, sfr, &sfr_free, poses, pose_free,
                          pts, pt_free, pix, of, oc, op};
  const CasparMojoOptions opt = {0, 200, 1.0, 1e-12, 2.0, 0.333333, 1e3, 0.0, 1.0, 500, 0.1};
  float out_intr[12], out_sfr[7], out_poses[NF * 7], out_pts[NP * 3];
  const CasparMojoState st = {out_intr, out_sfr, out_poses, out_pts};
  CasparMojoStats s;
  char err[256] = "";
  int32_t rc = 0;
  for (int run = 0; run < 3; ++run) { /* later calls show the per-call cost without the first-call start-up */
    rc = caspar_mojo_solve(&pr, &opt, &st, &s, err, sizeof(err));
    printf("run %d total %.4f s\n", run, s.total_seconds);
  }
  printf("abi %d rc %d cost %.6g -> %.6g iters %ld pcg %ld exit %ld solve %.4f s total %.4f s err '%s'\n",
         caspar_mojo_abi_version(), rc, s.initial_cost, s.final_cost, (long)s.iterations, (long)s.pcg_iterations,
         (long)s.exit_reason, s.solve_seconds, s.total_seconds, err);
  const int solved = rc == CASPAR_MOJO_OK && s.final_cost < 1e-6 * s.initial_cost && out_poses[3] == 1.0f;
  pr.camera_model = CASPAR_MOJO_FULL_OPENCV + 1;
  rc = caspar_mojo_solve(&pr, &opt, &st, &s, err, sizeof(err));
  printf("unsupported model: rc %d err '%s'\n", rc, err);
  return solved && rc == CASPAR_MOJO_UNSUPPORTED && strstr(err, "camera model") != NULL ? 0 : 1;
}
