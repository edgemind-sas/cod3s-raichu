//! A differential bench against the truth table: random coherent trees of
//! up to 13 events grouped in modules, with shared gates, constants and
//! votes, quantified under random engines, budgets and cutoffs. Written as
//! an independent review's fuzzer (300 000 trees, no disagreement once the
//! cut-set engine deduplicated and absorbed), kept here at 3 000.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    missing_docs,
    clippy::all
)]

use raichu_fta::{quantify, BasicEvent, Engine, Formula, GateDef, Law, QuantifySettings, Tree};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// Random coherent formula over events in `pool`, with sharing via gates.
fn formula(r: &mut Rng, pool: &[usize], depth: usize, gates: &[usize], consts: bool) -> Formula {
    if depth == 0 || r.below(4) == 0 {
        let c = r.below(20);
        if consts && c == 0 {
            return Formula::Constant {
                value: r.below(2) == 0,
            };
        }
        if !gates.is_empty() && c < 4 {
            return Formula::Gate {
                gate: gates[r.below(gates.len())],
            };
        }
        return Formula::event(pool[r.below(pool.len())]);
    }
    let n = 1 + r.below(4);
    let args: Vec<Formula> = (0..n)
        .map(|_| formula(r, pool, depth - 1, gates, consts))
        .collect();
    match r.below(3) {
        0 => Formula::and(args),
        1 => Formula::or(args),
        _ => {
            let k = 1 + r.below(args.len());
            Formula::at_least(k, args)
        }
    }
}

fn eval(f: &Formula, t: &Tree, x: &[bool]) -> bool {
    match f {
        Formula::Constant { value } => *value,
        Formula::Event { event } => x[*event],
        Formula::Gate { gate } => eval(&t.gates[*gate].formula, t, x),
        Formula::And { args } => args.iter().all(|a| eval(a, t, x)),
        Formula::Or { args } => args.iter().any(|a| eval(a, t, x)),
        Formula::AtLeast { k, args } => args.iter().filter(|a| eval(a, t, x)).count() >= *k,
        Formula::Not { arg } => !eval(arg, t, x),
    }
}

fn brute(t: &Tree, p: &[f64]) -> f64 {
    let n = p.len();
    let mut s = 0.0;
    for mask in 0u32..(1 << n) {
        let x: Vec<bool> = (0..n).map(|i| mask & (1 << i) != 0).collect();
        if eval(&t.top, t, &x) {
            s += (0..n)
                .map(|i| if x[i] { p[i] } else { 1.0 - p[i] })
                .product::<f64>();
        }
    }
    s
}

#[test]
fn every_engine_and_cutoff_agrees_with_the_truth_table() {
    let iters: u64 = 3_000;
    let seed0: u64 = 1;
    let mut fails = 0;
    let mut approx_under_bdd = 0;
    for it in 0..iters {
        let mut r = Rng(seed0.wrapping_mul(0x9E3779B97F4A7C15)
            ^ (it + 1).wrapping_mul(0xD1B54A32D192ED03)
            | 1);
        let n = 3 + r.below(11); // up to 13 events
        let consts = r.below(3) == 0;
        // Partition events into groups -> modules; top combines them.
        let ngroups = 1 + r.below(4);
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); ngroups];
        for e in 0..n {
            groups[r.below(ngroups)].push(e);
        }
        groups.retain(|g| !g.is_empty());
        let mut gates: Vec<GateDef> = Vec::new();
        let mut subs: Vec<Formula> = Vec::new();
        for g in &groups {
            // a couple of shared gates inside the group
            let mut local: Vec<usize> = Vec::new();
            for _ in 0..r.below(3) {
                let f = formula(&mut r, g, 3, &local, consts);
                gates.push(GateDef {
                    name: format!("G{}", gates.len()),
                    formula: f,
                });
                local.push(gates.len() - 1);
            }
            subs.push(formula(&mut r, g, 4, &local, consts));
        }
        // Nest: top over subs, possibly with a nested module chain
        let top = if subs.len() == 1 {
            subs.pop().unwrap()
        } else {
            let mut acc = subs.pop().unwrap();
            while let Some(s) = subs.pop() {
                acc = match r.below(3) {
                    0 => Formula::and(vec![acc, s]),
                    1 => Formula::or(vec![acc, s]),
                    _ => Formula::at_least(1 + r.below(2), vec![acc, s]),
                };
            }
            acc
        };
        let p: Vec<f64> = (0..n)
            .map(|_| match r.below(3) {
                0 => r.f() * 1e-3,
                1 => r.f(),
                _ => 0.5 + 0.5 * r.f(),
            })
            .collect();
        let t = Tree {
            name: "t".into(),
            events: p
                .iter()
                .enumerate()
                .map(|(i, q)| BasicEvent {
                    name: format!("E{i}"),
                    law: Law::Constant { probability: *q },
                })
                .collect(),
            gates,
            top,
        };
        let truth = brute(&t, &p);
        let engine = if r.below(2) == 0 {
            Engine::Auto
        } else {
            Engine::CutSets
        };
        let settings = QuantifySettings {
            engine,
            max_bdd_nodes: 2 + r.below(40),
            max_order: if r.below(2) == 0 {
                Some(r.below(4))
            } else {
                None
            },
            min_cut_probability: [0.0, 1e-4, 1e-2, 0.1, 0.3, 0.6][r.below(6)],
            max_cut_sets: 1 + r.below(6),
            max_expansions: [1u64, 2, 3, 5, 10, 30, 1000, 200_000][r.below(8)],
            cut_sets: r.below(2) == 0,
            ..Default::default()
        };
        let res = match std::panic::catch_unwind(|| quantify(&t, &settings)) {
            Ok(Ok(res)) => res,
            Ok(Err(e)) => panic!("case {it}: {e}"),
            Err(_) => {
                println!("PANIC it={it}");
                fails += 1;
                continue;
            }
        };
        if res
            .provenance
            .modules
            .iter()
            .any(|m| m.method == "bdd" && !m.exact)
        {
            approx_under_bdd += 1;
        }
        match res.upper_bound {
            Some(u) if u < truth - 1e-12 => {
                fails += 1;
                if fails <= 5 {
                    println!("FAIL it={it} upper={u} truth={truth} settings={settings:?}\n tree={t:#?}\n p={p:?}\n res={res:#?}");
                }
            }
            None => panic!("case {it}: no certified bound on a coherent tree"),
            _ => {}
        }
        if res.exact && (res.probability - truth).abs() > 1e-9 {
            println!("EXACT MISMATCH it={it}");
            fails += 1;
        }
        // full run w/o cutoffs: pivotal >= truth
        let full = QuantifySettings {
            engine,
            max_bdd_nodes: settings.max_bdd_nodes,
            ..Default::default()
        };
        if let Ok(f) = quantify(&t, &full) {
            if f.probability < truth - 1e-12 {
                println!("PIVOTAL BELOW it={it} {} {}", f.probability, truth);
                fails += 1;
            }
            if let Some(u) = f.upper_bound {
                if u < truth - 1e-12 {
                    println!("FULL UPPER BELOW it={it}");
                    fails += 1;
                }
            }
            let ex = quantify(
                &t,
                &QuantifySettings {
                    engine: Engine::Exact,
                    ..Default::default()
                },
            )
            .unwrap();
            if f.minimal_cut_sets != ex.minimal_cut_sets
                && f.minimal_cut_sets.is_some()
                && ex.minimal_cut_sets.is_some()
            {
                fails += 1;
                if fails <= 8 {
                    println!(
                        "CUTSETS DIFFER it={it} got={:?} exact={:?}",
                        f.minimal_cut_sets, ex.minimal_cut_sets
                    );
                }
            }
            // Omitted (None) is allowed under a tiny budget; a different
            // number never is.
            if let (Some(a), Some(b)) = (f.cut_set_count, ex.cut_set_count) {
                if f.cut_sets_complete && a != b {
                    println!("COUNT DIFFER it={it} {a} {b}");
                    fails += 1;
                }
            }
        }
    }
    assert_eq!(fails, 0, "{fails} disagreements with the truth table");
    // The bench must reach its purpose: approximated modules read by
    // exact parents.
    assert!(approx_under_bdd > 50, "{approx_under_bdd}");
}
