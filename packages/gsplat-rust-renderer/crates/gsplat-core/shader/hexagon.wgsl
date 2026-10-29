// Regular hexagon shared by 3D hexels and 2D fits: circumradius of a unit apothem, and the signed
// distance with flat top and bottom edges as the max of three half-plane distances.
const HEX_RADIUS = 1.15470053838;

fn hexagon_sdf(s: vec2f) -> f32 {
    let a = 0.5 * 1.7320508;
    return max(abs(s.y), max(abs(a * s.x + 0.5 * s.y), abs(a * s.x - 0.5 * s.y))) - 1.0;
}
