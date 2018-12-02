use rand::prelude::*;

pub fn one_d_kmeans(x: Vec<f64>, k:usize, iters: usize) -> Vec<usize> {
    let mut multid_form: Vec<Vec<f64>> = Vec::new();
    for e in x {
        multid_form.push(vec![e]);
    }
    nd_kmeans(multid_form, k, iters)
}

pub fn nd_kmeans(mut x: Vec<Vec<f64>>, k: usize, iters: usize) -> Vec<usize> {
    let dimensions = x[0].len();
    let mut rng = rand::thread_rng();
    x.partial_shuffle(&mut rng, k);
    let mut centroids = vec![Vec::new() as Vec<f64>; k];
    for i in 0..k {
        for j in 0..dimensions {
            centroids[i].push(x[i][j]);
        }
    }
    let mut centroid_assignments: Vec<usize> = vec![0; x.len()];
    let mut centroid_dists = vec![0.0; x.len()];
    let mut points_per_cluster = vec![0; centroids.len()];
    let mut next_centroids = vec![vec![0.0; dimensions]; k];
    
    points_per_cluster[0] = x.len(); // All points initially in cluster 0
    for m in 1..centroids.len() {
        points_per_cluster[m] = 0;
    }
    for _ in 0..iters {
        for j in 0..x.len() {
            for m in 0..centroids.len() {
                let mut new_dist = 0.0;
                for s in 0..dimensions {
                    new_dist += (x[j][s] - centroids[m][s]).powi(2);
                }
                if new_dist < centroid_dists[j] {
                    centroid_dists[j] = new_dist;
                    points_per_cluster[centroid_assignments[j]] -= 1;
                    centroid_assignments[j] = m;
                    points_per_cluster[m] += 1;
                }
            }
            for q in 0..dimensions {
                next_centroids[centroid_assignments[j]][q] += x[j][q];
            }
        }
        for m in 0..centroids.len() {
            for q in 0..dimensions {
                centroids[m][q] = next_centroids[m][q] / points_per_cluster[m] as f64; // Average the sum of positions
                next_centroids[m][q] = 0.0;
            }
        }
    }
    centroid_assignments
}