use std::ops;
use std::fmt;
use rand::prelude::*;
use rand::distributions::StandardNormal;
use std::cmp;

#[derive(Clone)]
pub struct Matrix {
    pub r: usize,
    pub c: usize,
    elements: Vec<Vec<f64>>
}

impl Matrix {
    pub fn zeros(r: usize, c: usize) -> Matrix {
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

    pub fn gaussian(r: usize, c: usize) -> Matrix {
        let mut rng = rand::thread_rng();

        let mut mat = Matrix::zeros(r, c);
        for i in 0..r {
            for j in 0..c {
                mat[i][j] = rng.sample(StandardNormal);
            }
        }
        mat
    }

    // Produce a matric from v*v.transpose()
    pub fn from_vec(v: &Vec<f64>) -> Matrix {
        let mut mat = Matrix::zeros(v.len(), v.len());
        for i in 0..v.len() {
            for j in 0..v.len() {
                mat[i][j] = v[i] * v[j];
            }
        }
        mat
    }

    pub fn identity(n: usize) -> Matrix {
        let mut mat = Matrix::zeros(n, n);
        for i in 0..n {
            mat[i][i] = 1.0;
        }
        mat
    }

    pub fn from_elements(elements: Vec<Vec<f64>>) -> Result<Matrix, &'static str> {
        let r = elements.len();
        let mut c = 0;
        if r > 0 {
            c = elements[0].len();
            for i in 1..r {
                if elements[i].len() != c {
                    return Err("Matrix must be rectangular");
                }
            }
        }
        Ok(Matrix {
            r: r,
            c: c,
            elements: elements
        })
    }

    // Zero indexed, inclusive start, non-inclusive end
    pub fn section(&self, rows: (usize, usize), columns: (usize, usize)) -> Matrix {
        let mut mat = Matrix::zeros(rows.1 - rows.0, columns.1 - columns.0);
        for i in 0..mat.r {
            for j in 0..mat.c {
                mat[i][j] = self[rows.0 + i][columns.0 + j];
            }
        }
        mat
    }

    pub fn extract_diagonal(&self) -> Vec<f64> {
        let mut d: Vec<f64> = Vec::new();
        for i in 0..cmp::min(self.r, self.c) {
            d.push(self[i][i]);
        }
        d
    }

    pub fn extract_column(&self, c: usize) -> Vec<f64> {
        let mut col: Vec<f64> = Vec::new();
        for i in 0..self.r {
            col.push(self[i][c]);
        }
        col
    }

    pub fn substitute(&mut self, sub: Matrix, row: usize, column: usize) {
        for i in 0..sub.r {
            for j in 0..sub.c {
                self[row + i][column + j] = sub[i][j];
            }
        }
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

    pub fn scale(&mut self, k: f64) {
        for i in 0..self.r {
            for j in 0..self.c {
                self[i][j] *= k;
            }
        }
    }

    pub fn qr_factorize(&mut self) -> Vec<f64> {
        let mut bs: Vec<f64> = Vec::new();
        for j in 0..self.c {
            let mut h: Vec<f64> = Vec::new();
            for i in j..self.r {
                h.push(self[i][j]);
            }
            let (v, b) = householder_vec(h);
            bs.push(b);
            let mut rep = Matrix::from_vec(&v);
            rep.scale(b);
            rep = (&Matrix::identity(self.r-j) - &rep).unwrap();
            rep = (&rep * &self.section((j, self.r), (j, self.c))).unwrap();
            self.substitute(rep, j, j);
            if j < self.r - 1 {
                for i in 0..self.r-j-1 {
                    self[j+1+i][j] = v[1+i];
                }
            }
        }
        bs
    }

    // Computes R*Q in place and returns Q
    fn rq_from_qr(&mut self, b: Vec<f64>) -> Matrix {
        if self.r != self.c {
            panic!("rq_from_qr is not yet implemented for non square matrices")
        }
        // Find Q by backwards accumulation
        let mut q = Matrix::identity(self.c);
        for bj in 0..self.c {
            let j = self.c - 1 - bj;
            let mut v = vec![0.0; self.c - j];
            v[0] = 1.0;
            for i in 1..v.len() {
                v[i] = self[j+i][j];
            }
            let mut rep = Matrix::from_vec(&v);
            rep.scale(b[j]);
            rep = (&Matrix::identity(self.c - j) - &rep).expect("Subtraction in rq_from_qr");
            let prod = (&rep * &q.section((j, self.c), (j, self.c))).expect("Matrix product iteration in rq_from_qr");
            q.substitute(prod, j, j);
        }

        let mut r = Matrix::zeros(self.c, self.c);
        for i in 0..self.r {
            for j in i..self.c {
                r[i][j] = self[i][j];
            }
        }
        let m = (&r * &q).expect("Failed R*Q");
        *self = m;
        q
    }

    // Returns eigenvalues and matrix of eigenvectors
    pub fn qr_iter(&mut self, k: usize) -> (Vec<f64>, Matrix) {
        let mut q = Matrix::identity(self.r);
        for _ in 0..k {
            let b = self.qr_factorize();
            q = (&q * &self.rq_from_qr(b)).expect("Failed multiplying Q1**Qn=Q");
        }
        (self.extract_diagonal(), q)
    }

    pub fn householder_bidiagonalize(&mut self) -> Vec<f64> {
        let mut bs: Vec<f64> = Vec::new();
        for j in 0..self.c {
            let mut h: Vec<f64> = Vec::new();
            for i in j..self.r {
                h.push(self[i][j]);
            }
            let (v, b) = householder_vec(h);
            bs.push(b);
            let mut rep = Matrix::from_vec(&v);
            rep.scale(b);
            rep = (&Matrix::identity(self.r-j) - &rep).unwrap();
            rep = (&rep * &self.section((j, self.r), (j, self.c))).unwrap();
            self.substitute(rep, j, j);
            for i in 0..self.r-j-1 {
                self[j+1+i][j] = v[1+i];
            }
            if j <= self.c - 2 {
                h = Vec::new();
                for k in j+1..self.c {
                    h.push(self[j][k]);
                }
                let (v, b) = householder_vec(h);
                let mut rep = Matrix::from_vec(&v);
                rep.scale(b);
                rep = (&Matrix::identity(self.c-j-1) - &rep).unwrap();
                rep = (&self.section((j, self.r), (j+1, self.c)) * &rep).unwrap();
                self.substitute(rep, j, j+1);
                for k in 0..self.c-j-2 {
                    self[j][j+2+k] = v[1+k];
                }
            }
        }
        bs
    }

    pub fn compute_svd(&mut self) -> Matrix {
        let b = self.householder_bidiagonalize();
        // Find U by backwards accumulation
        let mut u = Matrix::identity(self.r);
        for bj in 0..self.c {
            let j = self.c - 1 - bj;
            let mut v = vec![0.0; self.r - j];
            v[0] = 1.0;
            for i in 1..v.len() {
                v[i] = self[j+i][j];
            }
            let mut rep = Matrix::from_vec(&v);
            rep.scale(b[j]);
            rep = (&Matrix::identity(self.r - j) - &rep).expect("Subtraction in rq_from_qr");
            let prod = (&rep * &u.section((j, self.r), (j, self.r))).expect("Matrix product iteration in rq_from_qr");
            u.substitute(prod, j, j);
        }

        let mut b_sq = Matrix::zeros(self.c, self.c);
        for i in 0..self.c {
            b_sq[i][i] = self[i][i];
            if i < self.c - 1 {
                b_sq[i][i+1] = self[i][i+1];
            }
        }
        let (_, u_upper) = b_sq.qr_iter(3);
        let mut y = Matrix::zeros(self.r, self.c);
        y.substitute(u_upper, 0, 0);
        (&u * &y).unwrap()
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

pub fn householder_vec(x: Vec<f64>) -> (Vec<f64>, f64) {
    let mut s = 0.0;
    for i in 1..x.len() {
        s += x[i] * x[i];
    }
    let mut v = x.clone();
    v[0] = 1.0;
    if s == 0.0 {
        (v, 0.0)
    } else {
        let u = (x[0].powi(2) + s).sqrt();
        if x[0] <= 0.0 {
            v[0] = x[0] - u;
        } else {
            v[0] = -s / (x[0] + u);
        }
        let b = 2.0 * v[0].powi(2) / (s + v[0].powi(2));
        let vl = v.len();
        for i in 0..vl {
            v[vl - 1 - i] /= v[0];
        }
        (v, b)
    }
}

pub fn golub_kahan_svd_iteration(d: Vec<f64>, f: Vec<f64>) {
    // Check for 0's along diagonal and superdiagonal to break into subproblems
    // Zeros along diagonal are defined as ocurring when abs(d) < 0.000_001
    // TODO: this is an arbitrary choice of smallness, redefine to be consonant
    //      with machine precision * some matrix norm (of B)
    // Zeros along superdiagonal are defined as occurring when abs(f_i) < 0.000_000_1 * (abs(d_i) + abs(d_i+1))
    // Zeros along diagonal cause superdiagonal to be automatically zeroed
    let mut d_left: Vec<f64> = Vec::new(); // Diagonal
    let mut f_left: Vec<f64> = Vec::new(); // Superdiagonal
    let mut d_right: Vec<f64> = Vec::new(); // Diagonal
    let mut f_right: Vec<f64> = Vec::new(); // Superdiagonal

    let mut coupled = false;
    for i in 0..d.len() {
        if !coupled {
            if d[i].abs() < 0.000_001 {
                f_left.pop(); // Zero column so we have i-1 and i-2 items in d and f respectively
                coupled = true;
            } else {
                let mut nearzero = 0.000_000_1 * d[i].abs();
                if i < d.len() - 1 {
                    nearzero *= d[i+1];
                }
                if f[i].abs() < nearzero {
                    coupled = true;
                    d_right.push(d[i]);
                    if i < d.len()-1 {
                        f_right.push(f[i]);
                    }
                }
                d_left.push(d[i]);
                if i < d.len()-1 {
                    f_left.push(f[i]);
                }
            }
        } else {
            d_right.push(d[i]);
            if i < d.len()-1 {
                f_right.push(f[i]);
            }
        }
    }
    if d_left.len() != d.len() {
        golub_kahan_svd_iteration(d_right, f_right);
    }

}