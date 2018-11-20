use std::env;

mod graph;
mod matrix;

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args[1].trim().parse().expect("First argument must be a number");
    let g = graph::create_bipartite(n/2);
    for (i, d) in g.degrees().iter().enumerate() {
        println!("Degree of {} = {}", i, d);
    }
    println!("{}", g);
    println!("Laplacian:");
    println!("{}", g.laplacian());
    println!("{}", g.degree_norm());
}
