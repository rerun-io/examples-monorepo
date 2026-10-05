//! Rows shared by replay inputs.

#[derive(Debug, PartialEq)]
pub struct ImuRow {
    pub t_ns: i64,
    pub gyro: [f64; 3],
    pub accel: [f64; 3],
}
