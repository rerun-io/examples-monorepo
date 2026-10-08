## Camera format crosswalk

A subset fixes omitted coefficients to zero. A dash indicates no exact mapping provided here.

| Canonical family | COLMAP id/name and parameter order | OpenCV | Basalt | Kalibr | Aria | VSLAM-LAB |
|---|---|---|---|---|---|---|
| `pinhole` | 1 PINHOLE: fx,fy,cx,cy; 0 SIMPLE_PINHOLE: f,cx,cy | K, no distortion | pinhole | pinhole | — | pinhole |
| `brown_conrady`, Brown4 | 4 OPENCV: fx,fy,cx,cy,k1,k2,p1,p2 | D=[k1,k2,p1,p2] | pinhole-radtan8 subset | pinhole-radtan | — | radtan4 |
| `brown_conrady`, Brown5 | 6 FULL_OPENCV subset: fx,fy,cx,cy,k1,k2,p1,p2,k3,0,0,0 | D=[k1,k2,p1,p2,k3] | pinhole-radtan8 subset | — | — | radtan5 |
| `brown_conrady`, Brown8 | 6 FULL_OPENCV: fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6 | Rational D=[k1,k2,p1,p2,k3,k4,k5,k6] | pinhole-radtan8, plus fixed rpmax | — | — | radtan8 |
| `brown_conrady`, Brown12/14 | — | Brown8 plus s1,s2,s3,s4,tau_x,tau_y | — | — | Not Fisheye624 | — |
| `brown_conrady`, radial subsets | 2 SIMPLE_RADIAL: f,cx,cy,k1; 3 RADIAL: f,cx,cy,k1,k2 | Tangential terms zero | radtan8 subset | radtan subset | — | radtan4 subset |
| `kb4` | 5 OPENCV_FISHEYE: fx,fy,cx,cy,k1,k2,k3,k4 | fisheye D=[k1,k2,k3,k4] | kb4 | pinhole-equidistant | — | equid4 |
| `kb4`, radial subsets | 8 SIMPLE_RADIAL_FISHEYE: f,cx,cy,k1; 9 RADIAL_FISHEYE: f,cx,cy,k1,k2 | Remaining KB4 terms zero | kb4 subset | equidistant subset | — | equid4 subset |
| `fisheye624` | 11 RAD_TAN_THIN_PRISM_FISHEYE: fx,fy,cx,cy,k0..k5,p0,p1,s0..s3 | Neither Brown14 nor four-term fisheye | — | — | FISHEYE624 / FisheyeRadTanThinPrism; fx=fy | — |
| `fisheye624`, Fisheye62 subset | 11 with s0..s3=0 | — | — | — | Six angular and two tangential coefficients, prism zero | — |
| Unsupported thin-prism fisheye | 10 THIN_PRISM_FISHEYE: fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,sx1,sy1 | — | — | — | Not Fisheye624: distortion acts before radial scaling | — |
| Unsupported Apple radial magnification | No exact mapping | Brown8 fit is approximate | — | — | — | — |

COLMAP uses pixel centre (0.5,0.5) at the top left. These cameras use (0,0).
Import subtracts 0.5 from cx,cy; export adds 0.5. Brown p1 is the y-axis
term; Fisheye624 p_x (COLMAP p0) is the x-axis term. Basalt rpmax and
SDK visibility masks are separate domain metadata and cannot be discarded
by a lossless conversion.
