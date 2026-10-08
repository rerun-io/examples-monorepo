//! Time and check the cold fit (`cold::initial_pose`) on the golden set's cold rows, without Python.
//!
//! The fixture is written by `packages/handfit/tools/export_cold_golden.py` (its `--output`). Views are built from it exactly as
//! `handfit.HandFitter.fit` builds them, so the results are the Python binding's bit for bit.
//!
//! ```text
//! cargo run --release --example cold_bench -- <fixture> [--repeats N] [--stages] [--write results.txt] [--baseline results.txt]
//! ```
//!
//! `--write` stores every row's result; `--baseline` compares this run with a stored one (wrist, landmarks, energies, converged
//! flags, termination, hypothesis choice) and exits non-zero when a converged flag or the termination differs, or a wrist
//! moves more than 0.01 mm. `--stages` prints the time of the hypotheses, the 26 palm-only solves and the full solves (as
//! `initial_pose` reports them) and their terminations.
use handfit::cold::initial_pose;
use handfit::model::Step;
use handfit::nalgebra::{Matrix3, Matrix4, SMatrix, SVector, Vector2, Vector3};
use handfit::residual::cam_from_world;
use handfit::{Config, FitResult, JacobianMode, Model, Pose, View};
use std::collections::BTreeMap;
use std::time::Instant;

struct Row {
    index: usize,
    model: usize,
    mirror: f64,
    phi: f64,
    views: Vec<View>,
    torch: Pose,
    torch_energy: f64,
    torch_converged: bool,
}

/// One stored result: the pose and what the comparison needs.
struct Stored {
    translation: Vector3<f64>,
    rotation: Matrix3<f64>,
    landmarks: SVector<f64, 63>,
    energy: f64,
    converged: bool,
    termination: String,
    iterations: usize,
    chosen: Vec<usize>,
    winner: usize,
}

fn parse(words: &[&str]) -> Result<Vec<f64>, String> {
    words
        .iter()
        .map(|w| {
            w.parse::<f64>()
                .map_err(|e| format!("bad number {w:?}: {e}"))
        })
        .collect()
}

fn take<'a>(values: &'a [f64], offset: &mut usize, n: usize) -> Result<&'a [f64], String> {
    let slice = values
        .get(*offset..*offset + n)
        .ok_or_else(|| format!("record too short: need {} numbers", *offset + n))?;
    *offset += n;
    Ok(slice)
}

/// `HandFitter.add_camera` + `fit`: the view of one fixture record, in f64.
fn view(values: &[f64]) -> Result<View, String> {
    let mut o = 0;
    let cam_from_rig = Matrix4::from_row_slice(take(values, &mut o, 16)?);
    let transform = Matrix4::from_row_slice(take(values, &mut o, 16)?);
    let focal = Vector2::from_row_slice(take(values, &mut o, 2)?);
    let principal = Vector2::from_row_slice(take(values, &mut o, 2)?);
    let fisheye = take(values, &mut o, 1)?[0] != 0.0;
    let distortion = SVector::<f64, 8>::from_row_slice(take(values, &mut o, 8)?);
    let pixels = SMatrix::<f64, 21, 2>::from_row_slice(take(values, &mut o, 42)?);
    let weights = SVector::<f64, 21>::from_row_slice(take(values, &mut o, 21)?);
    let distances = SVector::<f64, 21>::from_row_slice(take(values, &mut o, 21)?);
    let (rotation, translation) = cam_from_world(&cam_from_rig, &transform);
    Ok(View {
        rotation,
        translation,
        camera: handfit::residual::camera_model(&focal, &principal, fisheye.then_some(&distortion)).map_err(|e| e.to_string())?,
        pixels,
        weights,
        distances,
    })
}

/// The fixture: the recorded fit config (phi per row), the hand models by id, the rows.
type Fixture = (Config, BTreeMap<usize, Model>, Vec<Row>);

fn load(path: &str) -> Result<Fixture, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut config = None;
    let mut models = BTreeMap::new();
    let mut rows: Vec<Row> = Vec::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let words: Vec<&str> = line.split_whitespace().collect();
        let values = parse(&words[1..])?;
        match words[0] {
            "config" => {
                // The binding's numbers with phi 1 in front (each row carries its own phi).
                let numbers: Vec<f64> = std::iter::once(1.0)
                    .chain(take(&values, &mut 0, 13)?.iter().copied())
                    .collect();
                config = Some(
                    Config::from_numbers(&numbers).map_err(|e| format!("config record: {e}"))?,
                );
            }
            "model" => {
                let mut o = 0;
                let id = take(&values, &mut o, 1)?[0] as usize;
                let model = Model {
                    axes: SMatrix::from_row_slice(take(&values, &mut o, 60)?),
                    pivots: SMatrix::from_row_slice(take(&values, &mut o, 60)?),
                    rest: SMatrix::from_row_slice(take(&values, &mut o, 63)?),
                    weights: SMatrix::from_row_slice(take(&values, &mut o, 63)?),
                    limits: SMatrix::from_row_slice(take(&values, &mut o, 40)?),
                };
                models.insert(id, model);
            }
            "row" => {
                let r = take(&values, &mut 0, 5)?;
                rows.push(Row {
                    index: r[0] as usize,
                    model: r[1] as usize,
                    mirror: if r[2] == 0.0 { 1.0 } else { -1.0 },
                    phi: r[3],
                    views: Vec::with_capacity(r[4] as usize),
                    torch: Pose {
                        rotation: Matrix3::identity(),
                        translation: Vector3::zeros(),
                        angles: SVector::zeros(),
                    },
                    torch_energy: f64::NAN,
                    torch_converged: false,
                });
            }
            "view" => rows
                .last_mut()
                .ok_or("view before row")?
                .views
                .push(view(&values)?),
            "torch" => {
                let row = rows.last_mut().ok_or("torch before row")?;
                let mut o = 0;
                row.torch.rotation = Matrix3::from_row_slice(take(&values, &mut o, 9)?);
                row.torch.translation = Vector3::from_row_slice(take(&values, &mut o, 3)?);
                row.torch.angles = SVector::from_row_slice(take(&values, &mut o, 22)?);
                row.torch_energy = take(&values, &mut o, 1)?[0];
                row.torch_converged = take(&values, &mut o, 1)?[0] != 0.0;
            }
            other => return Err(format!("unknown record {other:?}")),
        }
    }
    Ok((config.ok_or("no config record")?, models, rows))
}

fn quantile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.is_empty() {
        return f64::NAN;
    }
    // numpy's default linear interpolation.
    let position = q * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * (position - low as f64)
}

fn write(
    path: &str,
    rows: &[Row],
    results: &[(FitResult, SVector<f64, 63>)],
) -> Result<(), String> {
    let mut text = String::from(
        "# cold_bench results: row converged termination iterations energy winner chosen... | translation 3 rotation 9 landmarks 63\n",
    );
    for (row, (r, landmarks)) in rows.iter().zip(results) {
        let cold = r.cold.as_ref().ok_or("cold result without diagnostics")?;
        let chosen: Vec<String> = cold.chosen.iter().map(|c| c.to_string()).collect();
        let pose: Vec<String> = r
            .pose
            .translation
            .iter()
            .chain(r.pose.rotation.transpose().iter())
            .chain(landmarks.iter())
            .map(|x| format!("{x:?}"))
            .collect();
        text += &format!(
            "{} {} {} {} {:?} {} {} | {}\n",
            row.index,
            u8::from(r.converged),
            r.termination.as_str(),
            r.iterations,
            r.energies[3],
            cold.winner,
            chosen.join(" "),
            pose.join(" ")
        );
    }
    std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))
}

fn read_stored(path: &str) -> Result<BTreeMap<usize, Stored>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut stored = BTreeMap::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let (head, pose) = line.split_once(" | ").ok_or("result line without pose")?;
        let head: Vec<&str> = head.split_whitespace().collect();
        if head.len() < 6 {
            return Err(format!("short result line {line:?}"));
        }
        let numbers = |w: &str| w.parse::<f64>().map_err(|e| format!("{w:?}: {e}"));
        let pose = parse(&pose.split_whitespace().collect::<Vec<_>>())?;
        let mut o = 0;
        stored.insert(
            numbers(head[0])? as usize,
            Stored {
                converged: head[1] == "1",
                termination: head[2].to_string(),
                iterations: numbers(head[3])? as usize,
                energy: numbers(head[4])?,
                winner: numbers(head[5])? as usize,
                chosen: head[6..]
                    .iter()
                    .map(|w| numbers(w).map(|x| x as usize))
                    .collect::<Result<_, _>>()?,
                translation: Vector3::from_row_slice(take(&pose, &mut o, 3)?),
                rotation: Matrix3::from_row_slice(take(&pose, &mut o, 9)?),
                landmarks: SVector::from_row_slice(take(&pose, &mut o, 63)?),
            },
        );
    }
    Ok(stored)
}

fn landmark_error_mm(a: &SVector<f64, 63>, b: &SVector<f64, 63>) -> Vec<f64> {
    (0..21)
        .map(|i| (a.fixed_rows::<3>(3 * i) - b.fixed_rows::<3>(3 * i)).norm() * 1000.0)
        .collect()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let mut fixture = None;
    let mut repeats = 5;
    let mut stages = false;
    let mut write_path = None;
    let mut baseline = None;
    let mut i = 1;
    while i < args.len() {
        let next = |i: usize| {
            args.get(i + 1)
                .cloned()
                .ok_or(format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--repeats" => {
                repeats = next(i)?.parse().map_err(|e| format!("--repeats: {e}"))?;
                i += 1;
            }
            "--stages" => stages = true,
            "--write" => {
                write_path = Some(next(i)?);
                i += 1;
            }
            "--baseline" => {
                baseline = Some(next(i)?);
                i += 1;
            }
            path => fixture = Some(path.to_string()),
        }
        i += 1;
    }
    let fixture = fixture.ok_or(
        "usage: cold_bench <fixture> [--repeats N] [--stages] [--write results.txt] [--baseline results.txt]",
    )?;
    let (config, models, rows) = load(&fixture)?;
    let mode = JacobianMode::Analytic;
    let mut best = vec![f64::INFINITY; rows.len()];
    let mut stage_best = vec![[f64::INFINITY; 3]; rows.len()];
    let mut results: Vec<(FitResult, SVector<f64, 63>)> = Vec::new();
    let mut rigid_terminations: BTreeMap<&str, usize> = BTreeMap::new();
    let mut full_terminations: BTreeMap<&str, usize> = BTreeMap::new();
    for repeat in 0..repeats.max(1) {
        for (k, row) in rows.iter().enumerate() {
            let model = models
                .get(&row.model)
                .ok_or(format!("row {}: no model {}", row.index, row.model))?;
            let config = Config {
                phi: row.phi,
                ..config.clone()
            };
            let start = Instant::now();
            let result = initial_pose(model, &config, row.mirror, &row.views, mode)
                .map_err(|e| format!("row {}: {e}", row.index))?;
            best[k] = best[k].min(start.elapsed().as_secs_f64());
            if let Some(cold) = result.cold.as_ref() {
                let times = [cold.hypotheses_time, cold.rigid_time, cold.full_time];
                for (s, t) in times.into_iter().enumerate() {
                    stage_best[k][s] = stage_best[k][s].min(t.as_secs_f64());
                }
                if repeat == 0 {
                    for termination in &cold.rigid_terminations {
                        *rigid_terminations.entry(termination.as_str()).or_insert(0) += 1;
                    }
                    for termination in &cold.full_terminations {
                        *full_terminations.entry(termination.as_str()).or_insert(0) += 1;
                    }
                }
            }
            if repeat == 0 {
                let landmarks = model.landmarks(&result.pose, row.mirror, &Step::zeros(), None);
                results.push((result, landmarks));
            }
        }
    }
    let ms: Vec<f64> = best.iter().map(|t| t * 1000.0).collect();
    println!(
        "cold_bench {fixture}: {} cold rows, best of {repeats}; per cold fit median {:.3} ms, mean {:.3} ms, max {:.3} ms; total {:.2} ms",
        rows.len(),
        quantile(&ms, 0.5),
        ms.iter().sum::<f64>() / ms.len() as f64,
        quantile(&ms, 1.0),
        ms.iter().sum::<f64>()
    );
    if stages {
        let names = ["hypotheses", "palm-only solves", "full solves"];
        let line: Vec<String> = (0..3)
            .map(|s| {
                format!(
                    "{} {:.3} ms",
                    names[s],
                    stage_best.iter().map(|b| b[s] * 1000.0).sum::<f64>() / rows.len() as f64
                )
            })
            .collect();
        println!("  stage means: {}", line.join(", "));
        println!("  terminations: palm-only {rigid_terminations:?}, full {full_terminations:?}");
    }
    let rigid_iterations: Vec<usize> = results
        .iter()
        .flat_map(|(r, _)| {
            r.cold
                .as_ref()
                .map(|c| c.rigid_iterations.clone())
                .unwrap_or_default()
        })
        .collect();
    let full_iterations: Vec<usize> = results
        .iter()
        .flat_map(|(r, _)| {
            r.cold
                .as_ref()
                .map(|c| c.full_iterations.clone())
                .unwrap_or_default()
        })
        .collect();
    let at_cap = |v: &[usize]| v.iter().filter(|&&n| n >= config.init_iterations).count();
    println!(
        "  palm-only solves: {} ({:.1} iterations mean, {} at the {}-iteration cap); full solves: {} ({:.1} mean, {} at the cap)",
        rigid_iterations.len(),
        rigid_iterations.iter().sum::<usize>() as f64 / rigid_iterations.len().max(1) as f64,
        at_cap(&rigid_iterations),
        config.init_iterations,
        full_iterations.len(),
        full_iterations.iter().sum::<usize>() as f64 / full_iterations.len().max(1) as f64,
        at_cap(&full_iterations)
    );
    // Against the recorded torch results (handtrack's test_native_fit cold criteria).
    let mut translation_mm = Vec::new();
    let mut landmark_mm = Vec::new();
    let mut energy_delta = Vec::new();
    let mut torch_flips = 0;
    for (row, (r, landmarks)) in rows.iter().zip(&results) {
        let model = &models[&row.model];
        let torch_landmarks = model.landmarks(&row.torch, row.mirror, &Step::zeros(), None);
        translation_mm.push((r.pose.translation - row.torch.translation).norm() * 1000.0);
        landmark_mm.extend(landmark_error_mm(landmarks, &torch_landmarks));
        energy_delta.push(r.energies[3] as f32 as f64 - row.torch_energy);
        torch_flips += usize::from(r.converged != row.torch_converged);
    }
    println!(
        "  vs torch: wrist median {:.4} mm, p90 {:.4} mm, max {:.3} mm; landmarks median {:.4} mm, p99 {:.3} mm; rows with energy above torch {}; converged differs from torch on {} rows",
        quantile(&translation_mm, 0.5),
        quantile(&translation_mm, 0.9),
        quantile(&translation_mm, 1.0),
        quantile(&landmark_mm, 0.5),
        quantile(&landmark_mm, 0.99),
        energy_delta.iter().filter(|&&d| d > 0.0).count(),
        torch_flips
    );
    for (limit, q) in [(0.1, 0.5), (0.5, 0.9)] {
        if quantile(&translation_mm, q) > limit {
            for (row, (error, delta)) in rows.iter().zip(translation_mm.iter().zip(&energy_delta)) {
                if *error > limit && *delta > 0.0 {
                    println!(
                        "  FAIL vs torch: row {} wrist {error:.3} mm off with a higher energy (+{delta:.3e})",
                        row.index
                    );
                }
            }
        }
    }
    if let Some(path) = write_path {
        write(&path, &rows, &results)?;
        println!("  wrote {path}");
    }
    if let Some(path) = baseline {
        let stored = read_stored(&path)?;
        let mut wrist = Vec::new();
        let mut points = Vec::new();
        let mut rotation = Vec::new();
        let mut energy = Vec::new();
        let mut failures = Vec::new();
        let mut choice = 0;
        let mut iterations = 0;
        for (row, (r, landmarks)) in rows.iter().zip(&results) {
            let base = stored
                .get(&row.index)
                .ok_or(format!("baseline has no row {}", row.index))?;
            let cold = r.cold.as_ref().ok_or("cold result without diagnostics")?;
            let w = (r.pose.translation - base.translation).norm() * 1000.0;
            wrist.push(w);
            points.extend(landmark_error_mm(landmarks, &base.landmarks));
            rotation.push((r.pose.rotation - base.rotation).norm());
            energy.push((r.energies[3] - base.energy).abs() / base.energy.abs().max(1e-12));
            choice += usize::from(cold.chosen != base.chosen || cold.winner != base.winner);
            iterations += usize::from(r.iterations != base.iterations);
            if r.converged != base.converged
                || r.termination.as_str() != base.termination
                || w > 0.01
            {
                failures.push(format!(
                    "row {}: converged {} -> {}, termination {} -> {}, wrist moved {w:.4} mm",
                    row.index,
                    base.converged,
                    r.converged,
                    base.termination,
                    r.termination.as_str()
                ));
            }
        }
        println!(
            "  vs baseline {path}: wrist max {:.2e} mm, landmarks max {:.2e} mm, rotation max {:.2e}, energy max rel {:.2e}; \
             hypothesis choice differs on {choice} rows, iteration count on {iterations} rows",
            quantile(&wrist, 1.0),
            quantile(&points, 1.0),
            quantile(&rotation, 1.0),
            quantile(&energy, 1.0)
        );
        if !failures.is_empty() {
            for f in &failures {
                println!("  FAIL vs baseline: {f}");
            }
            return Err(format!("{} rows differ from the baseline", failures.len()));
        }
        println!(
            "  equivalent to the baseline: converged flags and terminations match on all {} rows",
            rows.len()
        );
    }
    Ok(())
}
