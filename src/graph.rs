use std::fmt;

use matrix::Matrix;

pub struct Graph {
    pub nodes: usize,
    pub adj: Matrix
}

impl Graph {
    pub fn new(n: usize) -> Graph {
        Graph {
            nodes: n,
            adj: Matrix::zeros(n)
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize, weight: Option<isize>) {
        let w = weight.unwrap_or(1);
        self.adj[u][v] = w;
        self.adj[v][u] = w;
    }

    pub fn degrees(&self) -> Vec<isize> {
        let mut deg: Vec<isize> = Vec::new();
        for row in &self.adj {
            let mut neighbors = 0;
            for col in row.iter() {
                if *col != 0 {
                    neighbors += 1;
                }
            }
            deg.push(neighbors);
        }
        deg
    }

    pub fn degree_matrix(&self) -> Matrix {
        Matrix::diagonal(self.degrees())
    }

    pub fn laplacian(&self) -> Matrix {
        (&self.degree_matrix() - &self.adj).expect("Failed to calculate laplacian")
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

