//! Sampling: repetition penalty -> temperature -> top-k -> top-p -> draw.

use rand::{Rng, SeedableRng, rngs::StdRng};

pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    rng: StdRng,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            repetition_penalty: 1.1,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// `history` is every token seen so far (prompt + generated), which is what
    /// HF's RepetitionPenaltyLogitsProcessor penalises.
    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        if self.repetition_penalty != 1.0 {
            let mut seen = std::collections::HashSet::new();
            for &id in history {
                if !seen.insert(id) {
                    continue;
                }
                let l = &mut logits[id as usize];
                if *l > 0.0 {
                    *l /= self.repetition_penalty;
                } else {
                    *l *= self.repetition_penalty;
                }
            }
        }

        // greedy
        if self.temperature <= 0.0 {
            return argmax(logits);
        }

        if self.temperature != 1.0 {
            let inv = 1.0 / self.temperature;
            for l in logits.iter_mut() {
                *l *= inv;
            }
        }
        softmax(logits);

        let n = logits.len();
        let mut order: Vec<usize> = (0..n).collect();
        let desc = |a: &usize, b: &usize| logits[*b].partial_cmp(&logits[*a]).unwrap();

        if self.top_k > 0 && self.top_k < n {
            order.select_nth_unstable_by(self.top_k - 1, desc);
            order.truncate(self.top_k);
        }
        order.sort_unstable_by(desc);

        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut cum = 0.0f32;
            let mut cut = order.len();
            for (i, &id) in order.iter().enumerate() {
                cum += logits[id];
                if cum > self.top_p {
                    cut = i + 1;
                    break;
                }
            }
            order.truncate(cut);
        }

        let total: f32 = order.iter().map(|&i| logits[i]).sum();
        if total <= 0.0 {
            return order[0] as u32;
        }
        // `gen` is a reserved keyword in edition 2024 (gen blocks), hence r#gen.
        let mut r: f32 = self.rng.r#gen();
        r *= total;
        for &id in &order {
            r -= logits[id];
            if r <= 0.0 {
                return id as u32;
            }
        }
        order[order.len() - 1] as u32
    }
}

fn argmax(x: &[f32]) -> u32 {
    let (idx, _) = x.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
        if *v > bv { (i, *v) } else { (bi, bv) }
    });
    idx as u32
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}
