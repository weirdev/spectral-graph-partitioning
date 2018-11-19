use std::ops;
use std::fmt;

pub struct Matrix {
    pub n: usize,
    elements: Vec<Vec<isize>>
}

impl Matrix {
    pub fn zeros(n: usize) -> Matrix{
        Matrix {
            n: n,
            elements: vec![vec![0; n]; n]
        }
    }

    pub fn diagonal(diagonals: Vec<isize>) -> Matrix {
        let mut mat = Matrix::zeros(diagonals.len());
        for i in 0..diagonals.len() {
            mat[i][i] = diagonals[i];
        }
        mat
    }
}

impl ops::Index<usize> for Matrix {
    type Output = Vec<isize>;

    fn index<'a>(&'a self, index: usize) -> &'a Self::Output {
        &self.elements[index]
    }
}

impl ops::IndexMut<usize> for Matrix {
    fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut Self::Output {
        &mut self.elements[index]
    }
}

impl<'a> ops::Sub for &'a Matrix {
    type Output = Result<Matrix, &'static str>;

    fn sub(self, other: &Matrix) -> Self::Output {
        if self.n != other.n {
            return Err("Subtraction on matrices must be on matrices of same dimensions")
        }
        let mut mat = Matrix::zeros(self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                mat[i][j] = self[i][j] - other[i][j];
            }
        }
        Ok(mat)
    }
}

impl IntoIterator for Matrix {
    type Item = Vec<isize>;
    type IntoIter = ::std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

impl<'a> IntoIterator for &'a Matrix {
    type Item = &'a Vec<isize>;
    type IntoIter = std::slice::Iter<'a, Vec<isize>>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements[..].iter()
    }
}

impl<'a> IntoIterator for &'a mut Matrix {
    type Item = &'a mut Vec<isize>;
    type IntoIter = std::slice::IterMut<'a, Vec<isize>>;

    fn into_iter(self) -> std::slice::IterMut<'a, Vec<isize>> {
        self.elements.iter_mut()
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for r in 0..self.n-1 {
            for c in &self[r] {
                write!(f, "{0: <5} ", c)?;
            }
            write!(f, "\n")?;
        }
        for c in &self[self.n - 1] {
            write!(f, "{: <5} ", c)?;
        }
        write!(f, "")
    }
}