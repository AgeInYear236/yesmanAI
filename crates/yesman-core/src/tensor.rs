use anyhow::{Result, anyhow};

/// Row-major (C-order) f32 tensor. Shapes are kept explicit so that mistakes in
/// weight layout surface as assertion failures instead of silent garbage.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(anyhow!("shape {shape:?} needs {n} elements, got {}", data.len()));
        }
        Ok(Self { data, shape })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![0.0; n], shape }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Interpret as a matrix `[rows, cols]`.
    pub fn as_matrix(&self) -> Result<(usize, usize)> {
        match self.shape.as_slice() {
            [r, c] => Ok((*r, *c)),
            other => Err(anyhow!("expected 2-D tensor, got {other:?}")),
        }
    }

    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    pub fn cols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1)
    }

    pub fn row(&self, i: usize) -> &[f32] {
        let c = self.cols();
        &self.data[i * c..(i + 1) * c]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_must_match_len() {
        assert!(Tensor::new(vec![0.0; 5], vec![2, 3]).is_err());
        assert!(Tensor::new(vec![0.0; 6], vec![2, 3]).is_ok());
    }
}
