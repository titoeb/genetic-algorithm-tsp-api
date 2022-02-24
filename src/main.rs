use genetic_algorithm_traits::Individual;
use genetic_algorithm_tsp::distance_mat;
use genetic_algorithm_tsp_api::tsp_solver;
use std::fs;
use std::time;

fn main() {
    // Load in the test matrix.
    let distances = distance_mat::DistanceMat::new(
        fs::read_to_string("tests/test-data/29_cities.txt")
            .unwrap()
            .lines()
            .collect::<Vec<&str>>()
            .iter()
            .map(|line| {
                line.split(";")
                    .map(|float_string| float_string.parse::<f64>().unwrap())
                    .collect::<Vec<f64>>()
            })
            .collect(),
    );
    // Get a solution
    let before = time::Instant::now();
    let best_invdividual = tsp_solver::solve_tsp(&distances, 600, 30, 10);
    let duration = tsp_solver::duration_to_ms(before.elapsed());
    let best_indiviudal_fitness = best_invdividual.fitness(&distances);

    println!(
        "The final individual was computed in {} ms and has a distance of {}",
        duration, -best_indiviudal_fitness
    );
    println!("The final individual is {:?}", best_invdividual);
}
