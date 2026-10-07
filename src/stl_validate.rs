//! Binary STL validation, shared by the `gen_dataset` and `reconstruct` binaries.
//!
//! Ported from the `validate_stl` helper in the original `tools/gen_dataset.py`.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

/// Validate that a binary STL is a closed, consistently oriented 2-manifold.
///
/// Returns the triangle count and the mesh extent.  This mirrors the checks in
/// the original Python tool so the Rust output can be held to the same standard.
pub fn validate_binary_stl(path: &Path) -> Result<(u32, [f64; 3]), String> {
    type Vtx = [u32; 3];
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    if data.len() < 84 {
        return Err("file too small for a binary STL".to_string());
    }
    let n = u32::from_le_bytes(data[80..84].try_into().unwrap());
    if data.len() != 84 + 50 * n as usize {
        return Err(format!("bad binary STL size: {} bytes", data.len()));
    }

    let mut edges: HashMap<(Vtx, Vtx), i64> = HashMap::new();
    let mut links: HashMap<Vtx, HashMap<Vtx, BTreeSet<Vtx>>> = HashMap::new();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];

    for i in 0..n as usize {
        let rec = &data[84 + 50 * i..84 + 50 * i + 50];
        let mut verts = [[0f32; 3]; 3];
        for (vi, vert) in verts.iter_mut().enumerate() {
            for (c, comp) in vert.iter_mut().enumerate() {
                let off = 12 + vi * 12 + c * 4;
                *comp = f32::from_le_bytes(rec[off..off + 4].try_into().unwrap());
            }
        }
        let v: [Vtx; 3] = verts.map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]);

        // reject zero-area triangles
        let (a, b, c) = (verts[0], verts[1], verts[2]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cross = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        if cross.iter().map(|x| x * x).sum::<f32>() < 1e-20 {
            return Err(format!("degenerate triangle at index {i}"));
        }

        for (x, y) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            *edges.entry((x, y)).or_insert(0) += 1;
        }
        for (x, y, z) in [(v[0], v[1], v[2]), (v[1], v[2], v[0]), (v[2], v[0], v[1])] {
            links.entry(x).or_default().entry(y).or_default().insert(z);
            links.entry(x).or_default().entry(z).or_default().insert(y);
        }

        for p in verts {
            for (axis, val) in p.iter().enumerate() {
                let val = *val as f64;
                lo[axis] = lo[axis].min(val);
                hi[axis] = hi[axis].max(val);
            }
        }
    }

    if let Some((e, count)) = edges.iter().find(|(_, c)| **c != 1) {
        return Err(format!(
            "non-manifold edge {e:?} appears {count} times (expected once)"
        ));
    }
    if let Some((e, _)) = edges
        .iter()
        .find(|((a, b), _)| !edges.contains_key(&(*b, *a)))
    {
        return Err(format!("open edge / inconsistent orientation: {e:?}"));
    }
    for (v, neighbours) in &links {
        if let Some((n, set)) = neighbours.iter().find(|(_, s)| s.len() != 2) {
            return Err(format!(
                "non-manifold vertex {v:?} (link degree {} for {n:?})",
                set.len()
            ));
        }
        let start = *neighbours.keys().next().unwrap();
        let mut seen: BTreeSet<Vtx> = BTreeSet::new();
        seen.insert(start);
        let mut stack = vec![start];
        while let Some(cur) = stack.pop() {
            for next in neighbours.get(&cur).into_iter().flatten() {
                if seen.insert(*next) {
                    stack.push(*next);
                }
            }
        }
        if seen.len() != neighbours.len() {
            return Err(format!("non-manifold vertex {v:?} (disconnected link)"));
        }
    }

    let extent = [0, 1, 2].map(|i| ((hi[i] - lo[i]) * 1000.0).round() / 1000.0);
    Ok((n, extent))
}
