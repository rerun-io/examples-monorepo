/**
 * dpretrieval -- pybind11 module wrapping DBoW2 for ORB-based image retrieval.
 *
 * Part of DPV-SLAM (Deep Patch Visual SLAM, Lipson et al., 2024).
 *
 * Provides the DPRetrieval class which maintains a DBoW2 vocabulary-tree
 * database.  Images are inserted via ORB feature detection; loop closure
 * candidates are identified by querying the database for visually similar
 * frames that are temporally distant (separated by at least `rad` frames).
 */

#include <pybind11/pybind11.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <thread>
#include <tuple>
#include <pybind11/stl.h>
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase
#include <pybind11/numpy.h>

using namespace DBoW2;
namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

/**
 * Reshape a cv::Mat of N rows into a vector of N single-row cv::Mat objects.
 * Required by the DBoW2 API which expects features as vector<cv::Mat>.
 */
void changeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
    out[i] = plain.row(i);
}

/**
 * Unused stub kept for API compatibility.  Originally intended for
 * estimating a 3D affine transform between two point sets, but the
 * loop closure pipeline uses RANSAC+Umeyama in Python instead.
 */
void estimateAffine3D(const py::array_t<uint8_t> &pointsA, const py::array_t<uint8_t> &pointsB){

        const py::buffer_info buf = pointsA.request();
        const cv::Mat lpts(buf.shape[0], 3, CV_64F, (double *)buf.ptr);
        std::cout << lpts << std::endl;

}

/** List of (train_x, train_y, query_x, query_y, distance) match tuples. */
typedef std::vector<std::tuple<double, double, double, double, double>> MatchList;

/**
 * ORB-based image retrieval using a DBoW2 vocabulary tree.
 *
 * Maintains an indexed database of ORB descriptors.  Each inserted image
 * is processed with OpenCV's ORB detector, and its descriptors are added
 * to the DBoW2 database for fast bag-of-words retrieval.  The query()
 * method finds the best-matching frame that is at least `rad` frames
 * away from the query frame (to avoid matching temporally adjacent frames).
 */
class DPRetrieval {
  private:
    std::vector<std::vector<cv::Mat > > features;
    std::vector<cv::Mat > descs;
    std::vector<std::vector<cv::KeyPoint > > kps;
    OrbDatabase db;
    const int rad;

  public:

    /**
     * Load an ORB vocabulary from a text file and initialize the database.
     *
     * @param vocab_path  Path to the ORB vocabulary file (e.g. "ORBvoc.txt").
     * @param rad         Minimum temporal distance between query and match
     *                    (frames closer than `rad` are ignored).
     */
    DPRetrieval(const std::string vocab_path, const int rad) : rad(rad){

      std::cout << "Loading the vocabulary " << vocab_path << std::endl;

      OrbVocabulary voc;
      voc.loadFromTextFile(vocab_path);

      db = OrbDatabase(voc, false, 0); // false = do not use direct index

    }

    /**
     * Detect ORB features in an image and add them to the database.
     *
     * @param array  Input image as a numpy uint8 array of shape (H, W, 3).
     */
    void insert_image(const py::array_t<uint8_t> &array){

        const py::buffer_info buf = array.request();
        if ((buf.shape.size() != 3) || buf.shape[2] != 3)
          throw std::invalid_argument( "invalid image shape" );

        const cv::Mat image(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

        cv::Mat mask;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        orb->detectAndCompute(image, mask, keypoints, descriptors);

        std::vector<cv::Mat > feats;
        changeStructure(descriptors, feats);
        kps.push_back(keypoints);
        features.push_back(feats);
        descs.push_back(descriptors);

        db.add(features.back());

    }

    /**
     * Brute-force match ORB descriptors between two indexed frames.
     *
     * @param ti  Index of the train (reference) frame.
     * @param qi  Index of the query frame.
     * @return    List of (train_x, train_y, query_x, query_y, distance) tuples.
     */
    MatchList match_pair(const int ti, const int qi) const {
      cv::BFMatcher matcher(cv::NORM_HAMMING, true);
      cv::Mat train_descriptors = descs.at(ti);
      cv::Mat query_descriptors = descs.at(qi);

      std::vector<std::vector<cv::DMatch>> knn_matches;
      matcher.knnMatch(query_descriptors, train_descriptors, knn_matches, 1);

      MatchList output;

      for (const auto &pair_list : knn_matches){
        if (!pair_list.empty()){
          const auto &pair = pair_list.back();

          auto trainpt = (kps[ti][pair.trainIdx].pt);
          auto querypt = (kps[qi][pair.queryIdx].pt);

          output.emplace_back(trainpt.x, trainpt.y, querypt.x, querypt.y, pair.distance);
        }
      }

      return output;
    }

    /**
     * Query the DBoW2 database for the best non-recent match.
     *
     * Returns the highest-scoring match that is at least `rad` frames away
     * from the query frame, along with the ORB keypoint correspondences.
     *
     * @param i  Index of the query frame (must have been previously inserted).
     * @return   Tuple of (score, matched_frame_index, matches).
     *           Returns (-1, -1, []) if no valid match is found.
     */
    auto query(const int i) const {

      if ((i >= features.size()) || (i < 0))
        throw std::invalid_argument( "index invalid" );

      QueryResults ret;
      db.query(features[i], ret, 4);
      std::tuple<float, int, MatchList> output(-1, -1, {});
      for (const auto &r : ret){
        int j = r.Id;
        if ((abs(j - i) >= rad) && (r.Score > std::get<0>(output))){
          const MatchList matches = match_pair(i, j);
          output = std::make_tuple(r.Score, j, matches);
        }
      }
      return output;

    }
};


PYBIND11_MODULE(dpretrieval, m) {
  m.doc() = "ORB-based image retrieval using DBoW2 vocabulary tree for loop closure detection.";

  py::class_<DPRetrieval>(m, "DPRetrieval")
    .def(py::init<std::string, int>())
    .def("insert_image", &DPRetrieval::insert_image)
    .def("match_pair", &DPRetrieval::match_pair)
    .def("query", &DPRetrieval::query);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
