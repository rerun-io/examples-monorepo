// Dump RootSLAM/Basalt BRIEF descriptors, Hamming matches and HashBoW queries for the kornia-staging golden tests.
//
// The output is Basalt's own: this file only feeds inputs to Basalt's computeAngles, computeDescriptors,
// matchDescriptors (src/utils/keypoints.cpp) and HashBowST (include/basalt/hash_bow/hash_bow.h) and prints what
// they return. The Rust golden tests read it from the NAS and skip when it is absent.
//
// Build and run against a RootSLAM checkout (branch pixi, built with `pixi run build`), with RS = that checkout:
//   RS/.pixi/envs/default/bin/c++ -std=c++17 -O3 -march=native -DBASALT_BUILD_SHARED_LIBRARY_ONLY \
//     -I RS/build/include -I RS/include -I RS/thirdparty/basalt-headers/include \
//     -isystem RS/thirdparty/basalt-headers/thirdparty/eigen -isystem RS/thirdparty/basalt-headers/thirdparty/Sophus \
//     -isystem RS/thirdparty/basalt-headers/thirdparty/cereal/include -isystem RS/.pixi/envs/default/include \
//     -isystem RS/.pixi/envs/default/include/opencv4 basalt_brief_dump.cpp RS/src/utils/keypoints.cpp \
//     -L RS/.pixi/envs/default/lib -Wl,-rpath,RS/.pixi/envs/default/lib -lopencv_core -lopencv_imgproc \
//     -lopencv_features2d -ltbb -o basalt_brief_dump
//   ./basalt_brief_dump packages/kornia-staging/fixtures/frames/frame_00{0,1,2}_cam{0,1}.pgm > basalt_brief.txt
//
// Inputs (all reproduced by the Rust test): each PGM widened to u16 with `v << 8`, then one synthetic u16 image
// (xorshift32 seeded 12345, `state >> 16`, then an integer 3x3 box mean). Keypoints on an 11-pixel grid with
// fractional offsets, so the 19-pixel border test is exercised on both sides.
#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <basalt/hash_bow/hash_bow.h>
#include <basalt/utils/keypoints.h>

namespace {

const int EDGE = 19;  // EDGE_THRESHOLD in keypoints.cpp, the border loop_closing.cpp:138 tests
const float FRACTIONS[5] = {0.0f, 0.25f, 0.5f, 0.999f, 0.37f};

std::vector<uint16_t> read_pgm(const std::string& path, int& w, int& h) {
  std::ifstream in(path, std::ios::binary);
  std::string magic;
  int maxval = 0;
  in >> magic >> w >> h >> maxval;
  in.get();
  std::vector<uint8_t> bytes(static_cast<size_t>(w) * h);
  in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  std::vector<uint16_t> out(bytes.size());
  for (size_t i = 0; i < bytes.size(); i++) out[i] = static_cast<uint16_t>(bytes[i] << 8);
  return out;
}

std::vector<uint16_t> synthetic(int w, int h) {
  std::vector<uint32_t> noise(static_cast<size_t>(w) * h);
  uint32_t state = 12345;
  for (auto& v : noise) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    v = state >> 16;
  }
  std::vector<uint16_t> out(noise.size());
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      uint32_t sum = 0, count = 0;
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
          int xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          sum += noise[static_cast<size_t>(yy) * w + xx];
          count++;
        }
      out[static_cast<size_t>(y) * w + x] = static_cast<uint16_t>(sum / count);
    }
  return out;
}

std::string hex(const std::bitset<256>& d) {
  // 4 little-endian u64 words, bit i of the bitset = bit i % 64 of word i / 64.
  std::string s;
  char buf[17];
  for (int word = 0; word < 4; word++) {
    uint64_t v = 0;
    for (int b = 0; b < 64; b++)
      if (d[word * 64 + b]) v |= uint64_t(1) << b;
    std::snprintf(buf, sizeof(buf), "%016" PRIx64, v);
    s += buf;
    if (word < 3) s += ' ';
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::vector<uint16_t>> images;
  std::vector<int> widths, heights;
  for (int a = 1; a < argc; a++) {
    int w = 0, h = 0;
    images.push_back(read_pgm(argv[a], w, h));
    widths.push_back(w);
    heights.push_back(h);
  }
  images.push_back(synthetic(320, 240));
  widths.push_back(320);
  heights.push_back(240);

  const size_t tiles = 4;
  basalt::HashBowST<256> hash_bow(16);
  std::map<basalt::TimeCamId, basalt::HashBowVector> bags;
  std::vector<std::vector<std::bitset<256>>> per_image;

  for (size_t k = 0; k < images.size(); k++) {
    const int w = widths[k], h = heights[k];
    basalt::Image<const uint16_t> img(images[k].data(), w, h, w * sizeof(uint16_t));
    std::printf("image %zu %d %d\n", k, w, h);
    basalt::KeypointsData kd;
    std::vector<std::pair<float, float>> all;
    std::vector<bool> inside;
    std::vector<size_t> tile_of;
    size_t n = 0;
    for (int gy = 0; gy * 11 < h; gy++)
      for (int gx = 0; gx * 11 < w; gx++, n++) {
        float x = gx * 11 + FRACTIONS[n % 5];
        float y = gy * 11 + FRACTIONS[(n / 5 + gx) % 5];
        all.emplace_back(x, y);
        bool in = img.InBounds(x, y, EDGE);
        inside.push_back(in);
        if (in) {
          kd.corners.emplace_back(Eigen::Vector2f(x, y).cast<double>());
          tile_of.push_back((static_cast<size_t>(y) * tiles / h) * tiles + static_cast<size_t>(x) * tiles / w);
        }
      }
    basalt::computeAngles(img, kd, true);
    basalt::computeDescriptors(img, kd);
    size_t j = 0;
    for (size_t i = 0; i < all.size(); i++) {
      if (!inside[i]) {
        std::printf("kp %.9g %.9g out\n", all[i].first, all[i].second);
        continue;
      }
      std::printf("kp %.9g %.9g %.17g %s\n", all[i].first, all[i].second, kd.corner_angles[j],
                  hex(kd.corner_descriptors[j]).c_str());
      j++;
    }
    per_image.push_back(kd.corner_descriptors);

    // One HashBoW entry per (image, tile): frame = tile * 10 + image / 2, cam = image % 2.
    for (size_t t = 0; t < tiles * tiles; t++) {
      std::vector<std::bitset<256>> descriptors;
      for (size_t i = 0; i < tile_of.size(); i++)
        if (tile_of[i] == t) descriptors.push_back(kd.corner_descriptors[i]);
      basalt::HashBowVector bow;
      std::vector<basalt::FeatureHash> hashes;
      hash_bow.compute_bow(descriptors, hashes, bow);
      basalt::TimeCamId tcid(static_cast<int64_t>(t * 10 + k / 2), k % 2);
      hash_bow.add_to_database(tcid, bow);
      bags[tcid] = bow;
      std::printf("bag %" PRId64 " %zu %zu", tcid.frame_id, tcid.cam_id, bow.size());
      std::map<uint32_t, double> sorted;
      for (const auto& [word, weight] : bow) sorted[static_cast<uint32_t>(word.to_ulong())] = weight;
      for (const auto& [word, weight] : sorted) std::printf(" %u:%.17g", word, weight);
      std::printf("\n");
    }
  }

  // Matches between consecutive images of equal size, Basalt's loop-closing thresholds.
  for (size_t k = 0; k + 1 < images.size(); k++) {
    if (widths[k] != widths[k + 1]) continue;
    std::vector<std::pair<int, int>> matches;
    basalt::matchDescriptors(per_image[k], per_image[k + 1], matches, 70, 1.2);
    std::printf("match %zu %zu %zu", k, k + 1, matches.size());
    std::map<int, int> sorted(matches.begin(), matches.end());
    for (const auto& [a, b] : sorted) std::printf(" %d:%d", a, b);
    std::printf("\n");
  }

  // Every bag queried: all results (unsorted, as returned), and the top 10 with the time filter frame < query - 5.
  for (const auto& [tcid, bow] : bags) {
    std::vector<std::pair<basalt::TimeCamId, double>> results;
    hash_bow.querry_database(bow, 1000000, results);
    std::printf("query_all %" PRId64 " %zu %zu", tcid.frame_id, tcid.cam_id, results.size());
    for (const auto& [id, score] : results) std::printf(" %" PRId64 ",%zu,%.17g", id.frame_id, id.cam_id, score);
    std::printf("\n");
    int64_t max_t_ns = tcid.frame_id - 5;
    hash_bow.querry_database(bow, 10, results, &max_t_ns);
    std::printf("query_top %" PRId64 " %zu %zu", tcid.frame_id, tcid.cam_id, results.size());
    for (const auto& [id, score] : results) std::printf(" %" PRId64 ",%zu,%.17g", id.frame_id, id.cam_id, score);
    std::printf("\n");
  }
  return 0;
}
