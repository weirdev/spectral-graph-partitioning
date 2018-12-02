use std::fmt;

use matrix::{Matrix, DiagonalMatrix};

pub struct Graph {
    pub nodes: usize,
    pub adj: Matrix
}

impl Graph {
    pub fn new(n: usize) -> Graph {
        Graph {
            nodes: n,
            adj: Matrix::zeros(n, n)
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize, weight: Option<f64>) {
        let w = weight.unwrap_or(1.0);
        self.adj[u][v] = w;
        self.adj[v][u] = w;
    }

    pub fn degrees(&self) -> Vec<f64> {
        let mut deg: Vec<f64> = Vec::new();
        for row in &self.adj {
            let mut neighbors = 0.0;
            for col in row.iter() {
                if *col != 0.0 {
                    neighbors += 1.0;
                }
            }
            deg.push(neighbors);
        }
        deg
    }

    pub fn degree_matrix(&self) -> DiagonalMatrix {
        DiagonalMatrix::new(self.degrees())
    }

    pub fn laplacian(&self) -> Matrix {
        (&self.degree_matrix().to_square() - &self.adj).expect("Failed to calculate laplacian")
    }

    pub fn degree_norm(&self) -> Matrix {
        let mut k = self.degree_matrix();
        k.ip_powf(-0.5);
        let mut d = self.degree_matrix();
        d.ip_powf(-0.5);
        d.left_mul_sq(&d.right_mul_sq(&self.adj).unwrap()).unwrap()
    }

    pub fn approx_k_eigenvecs(&self, k: usize) -> Matrix {
        let w = self.degree_norm();
        let w_t = w.transpose();
        let p = ((k * self.nodes) as f64).ln() as usize;
        let itr = (&w * &w_t).unwrap().powi(p).unwrap();
        // Multiply by S = nxk w/ gaussian elements
        let s = Matrix::gaussian(self.nodes, k);

        (&(&itr * &w).unwrap() * &s).unwrap()
    }

    pub fn qr_eigenvecs(&self) -> Vec<f64> {
        let mut mat = self.adj.clone();
        mat.qr_iter(5);
        mat.extract_diagonal()
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{} nodes", self.nodes)?;
        write!(f, "{}", self.adj)
    }
}

pub fn create_pairwise_connected(pairs: usize) -> Graph {
    let mut g = Graph::new(pairs*2);
    for i in 0..pairs {
        g.add_edge(i, i+pairs, None);
    }
    g
}

pub fn create_bipartite(half_size: usize) -> Graph {
    let mut g = Graph::new(half_size*2);
    for i in 0..half_size {
        for j in half_size..half_size*2 {
            g.add_edge(i, j, None);
        }
    }
    g
}

pub fn create_from_2d_points(x: Vec<f64>, y: Vec<f64>) -> Result<Graph, &'static str> {
    if x.len() != y.len() {
        return Err("x and y coordinate vecs must be of same length")
    }
    let variance = variance(&x)? + variance(&y)?;
    let doublevar = variance * 2.0;
    let mut g = Graph::new(x.len());
    for i in 0..x.len()-1 {
        for j in i+1..x.len() {
            let sqrdist = (x[i] - x[j]).powi(2) + (y[i] - y[j]).powi(2);
            let normalized = -1.0 * sqrdist / doublevar;
            g.add_edge(i, j, Some(normalized.exp()));
        }
    }
    Ok(g)
}

pub fn create_2fc_kconnections(fcsize: usize, k: usize) -> Graph {
    let mut g = Graph::new(fcsize*2);
    // FC subgraphs
    for i in 0..fcsize-1 {
        for j in i+1..fcsize {
            g.add_edge(i, j, None);
            g.add_edge(fcsize + i, fcsize + j, None);
        }
    }
    // K connection between FC layers
    for i in 0..k {
        g.add_edge(i, fcsize + i, None);
    }
    g
}

fn variance(x: &Vec<f64>) -> Result<f64, &'static str> {
    if x.len() == 0 {
        return Err("Attempted to take variance of empty vec")
    }
    let m = mean(x);
    let mut var: f64 = 0.0;
    for v in x {
        let diff = *v as f64 - m;
        var += diff * diff;
    }
    Ok(var / x.len() as f64)
}

fn mean(x: &Vec<f64>) -> f64 {
    if x.len() > 0 {
        let s: f64 = x.iter().sum();
        s / x.len() as f64
    }
    else {
        0.0
    }
}