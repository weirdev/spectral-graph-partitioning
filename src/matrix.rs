use std::ops;
use std::fmt;

#[derive(Clone)]
pub struct Matrix {
    pub r: usize,
    pub c: usize,
    elements: Vec<Vec<f64>>
}

impl Matrix {
    pub fn zeros(r: usize, c: usize) -> Matrix{
        Matrix {
            r: r,
            c: c,
            elements: vec![vec![0.0; c]; r]
        }
    }

    pub fn diagonal(diagonals: Vec<f64>) -> Matrix {
        let mut mat = Matrix::zeros(diagonals.len(), diagonals.len());
        for i in 0..diagonals.len() {
            mat[i][i] = diagonals[i];
        }
        mat
    }

    pub fn transpose(&self) -> Matrix {
        let mut mat = Matrix::zeros(self.c, self.r);
        for i in 0..self.r {
            for j in 0..self.c {
                mat[j][i] = self[i][j];
            }
        }
        mat
    }

    pub fn powi(&self, p: usize) -> Result<Matrix, &'static str> {
        if self.r != self.c {
            return Err("Matrix power operation must be performed on a square matrix")
        }
        if p == 0 {
            Ok(Matrix::diagonal(vec![1.0; self.r]))
        } else {
            let mut bin = binary_decomposition(p);
            bin.reverse();
            let mut prod = self.clone();
            for i in 1..bin.len() {
                prod = (&prod * &prod).unwrap();
                if bin[i] {
                    prod = (&prod * self).unwrap();
                }
            }
            Ok(prod)
        }
    }
}

pub struct DiagonalMatrix {
    pub n: usize,
    elements: Vec<f64>
}

impl DiagonalMatrix {
    pub fn new(diagonals: Vec<f64>) -> DiagonalMatrix {
        DiagonalMatrix {
            n: diagonals.len(),
            elements: diagonals
        }
    }

    pub fn to_square(self) -> Matrix {
        Matrix::diagonal(self.elements)
    }

    pub fn left_mul_sq(&self, lhs: &Matrix) -> Result<Matrix, String> {
        if self.n != lhs.c {
            return Err(format!("Incompatible dimensions ({}x{})*({2}x{2})  for matrix multiplications", lhs.r, lhs.c, self.n))
        }
        let mut mat = Matrix::zeros(lhs.r, self.n);
        for i in 0..lhs.r {
            for j in 0..self.n {
                mat[i][j] = lhs[i][j] * self.elements[j];
            }
        }
        Ok(mat)
    }

    pub fn right_mul_sq(&self, rhs: &Matrix) -> Result<Matrix, String> {
        if self.n != rhs.r {
            return Err(format!("Incompatible dimensions ({0}x{0})*({1}x{2})  for matrix multiplications", self.n, rhs.r, rhs.c))
        }
        let mut mat = Matrix::zeros(self.n, rhs.c);
        for i in 0..self.n {
            for j in 0..rhs.c {
                mat[i][j] = self.elements[i] * rhs[i][j];
            }
        }
        Ok(mat)
    }

    pub fn ip_powf(&mut self, p: f64) {
        for i in 0..self.elements.len() {
            self.elements[i] = self.elements[i].powf(p);
        }
    }
}

impl ops::Index<usize> for Matrix {
    type Output = Vec<f64>;

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
        if self.r != other.r || self.c != other.c {
            return Err("Subtraction on matrices must be on matrices of same dimensions")
        }
        let mut mat = Matrix::zeros(self.r, self.c);
        for i in 0..self.r {
            for j in 0..self.c {
                mat[i][j] = self[i][j] - other[i][j];
            }
        }
        Ok(mat)
    }
}

impl<'a> ops::Mul for &'a Matrix {
    type Output = Result<Matrix, String>;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        if self.c != rhs.r {
            return Err(format!("Incompatible dimensions ({}x{})*({}x{})  for matrix multiplications", self.r, self.c, rhs.r, rhs.c))
        }
        let mut mat = Matrix::zeros(self.r, rhs.c);
        for i in 0..self.r {
            for j in 0..rhs.c {
                let mut sum = 0.0;
                for x in 0..self.c {
                    sum += self[i][x] * rhs[x][j];
                }
                mat[i][j] = sum;
            }
        }
        Ok(mat)
    }
}

impl IntoIterator for Matrix {
    type Item = Vec<f64>;
    type IntoIter = ::std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

impl<'a> IntoIterator for &'a Matrix {
    type Item = &'a Vec<f64>;
    type IntoIter = std::slice::Iter<'a, Vec<f64>>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements[..].iter()
    }
}

impl<'a> IntoIterator for &'a mut Matrix {
    type Item = &'a mut Vec<f64>;
    type IntoIter = std::slice::IterMut<'a, Vec<f64>>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.iter_mut()
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for r in 0..self.r-1 {
            for c in &self[r] {
                write!(f, "{0: <5} ", c)?;
            }
            write!(f, "\n")?;
        }
        for c in &self[self.r - 1] {
            write!(f, "{: <5} ", c)?;
        }
        write!(f, "")
    }
}

fn binary_decomposition(mut x: usize) -> Vec<bool> {
    let mut bin: Vec<bool> = Vec::new();
    while x > 0 {
        if x & 1 == 1 {
            bin.push(true);
            x -= 1;
        } else {
            bin.push(false);
        }
        x /= 2;
    }
    bin
}