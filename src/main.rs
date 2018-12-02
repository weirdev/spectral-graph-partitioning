extern crate rand;

use std::env;

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
    for (i, d) in g.degrees().iter().enumerate() {
        println!("Degree of {} = {}", i, d);
    }
    println!("{}", g);
    //println!("{}", g.laplacian());
    //println!("{}", g.approx_k_eigenvecs(2));
    let mut adj = g.adj.clone();
    let (eigenvals, eigenvecs) = adj.qr_iter(3);
    println!("QR Algorithm\n{}\n{:?}\n{}", adj, eigenvals, eigenvecs);
}
