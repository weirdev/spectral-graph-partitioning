extern crate rand;

use rand::prelude::*;

use std::env;
use std::cmp::Ordering;

mod graph;
mod matrix;
mod cluster;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        panic!("Missing argument(s)");
    }
    let n: usize = args[1].trim().parse().expect("First argument must be a number");
    //let g = graph::create_bipartite(n/2);
    let g = graph::create_2fc_kconnections(n, 1);
    /*for (i, d) in g.degrees().iter().enumerate() {
        println!("Degree of {} = {}", i, d);
    }*/
    //println!("{}", g.laplacian());
    //println!("{}", g.laplacian());
    //println!("{}", g.approx_k_eigenvecs(2));
    //let mut adj = g.adj.clone();
    let mut lap = g.degree_normed_adj();
    let (eigenvals, eigenvecs) = lap.qr_iter(3);
    //println!("QR Algorithm\n{}\n{:?}\n{}", lap, eigenvals, eigenvecs);
    let clustered = spectral_cluster(eigenvals, eigenvecs);
    println!("Clustered\n{:?}", clustered);
}

pub fn spectral_cluster(eigenvals: Vec<f64>, eigenvecs: matrix::Matrix) -> Vec<usize> {
    let mut evi: Vec<_> = eigenvals.iter().enumerate().collect();
    evi.sort_by(|(_, ev1), (_, ev2)| ev1.partial_cmp(ev2).unwrap_or(Ordering::Equal));

    //println!("Second largest eigenvector\n{:?}", eigenvecs.extract_column(evi[evi.len()-2].0));
    // Cluster on second largest eigenvalue
    cluster::one_d_kmeans(eigenvecs.extract_column(evi[evi.len()-2].0), 2, 7)
}