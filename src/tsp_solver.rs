use genetic_algorithm_traits::Population;
use genetic_algorithm_tsp::{distance_mat, route};
use std::time;

/// From a `std::time::Duration` object compute the elapsed microseconds.
///
/// # Arguments
///
/// * `duration` - The time that has gone by.
///
/// # Examples
///
/// ```
/// use std::time;
/// use std::thread;
/// use genetic_algorithm_tsp_api::tsp_solver;
///
/// let before = time::Instant::now();
/// thread::sleep(time::Duration::from_millis(10));
/// println!("Sleeping for 10 ms took {} ms", tsp_solver::duration_to_ms(before.elapsed()));
/// ```
pub fn duration_to_ms(duration: time::Duration) -> u64 {
    let nano_seconds = duration.subsec_nanos() as u64;
    (1000 * 1000 * 1000 * duration.as_secs() + nano_seconds) / (1000 * 1000)
}

/// Compute an route that for the traveling-salesman-problem defined by
/// the distance matrix.
///
/// # Arguments
///
/// * `distance_matrix` - These distances define the fitness of an invidual.
/// * `n_generation` - How many generations should the algorithm run for?
/// * `n_routes` - How many routes should be kept in the population.
/// * `n_random_route_per_generation` - How many random routes should be
///     ingested in every generation to allow?
pub fn solve_tsp(
    distance_matrix: &distance_mat::DistanceMat,
    n_generations: usize,
    n_routes: usize,
    n_random_individuals_per_generation: usize,
    top_n: usize,
) -> Vec<route::Route> {
    let initial_population = distance_matrix.get_random_population(n_routes);
    // Decay mutation probability.
    (0..10000)
        .step_by(10000 / n_generations)
        .fold(
            initial_population,
            |population, mutation_probability_int| {
                population
                    .evolve(1.0 - (f64::from(mutation_probability_int) / 10000.0) as f32)
                    // Add a few random inidividuals each round.
                    .add_n_random_nodes(n_random_individuals_per_generation)
                    .get_fittest_population(n_routes, distance_matrix)
            },
        )
        .get_n_fittest(top_n, distance_matrix)
}

mod tests {
    #[test]
    fn test_duration() {
        use super::duration_to_ms;
        use std::thread;
        use std::time;

        // Test that after waiting for 12 ms, `duration_to_ms`
        // reports correctly that 12 ms have gone by.
        let before = time::Instant::now();
        thread::sleep(time::Duration::from_millis(12));
        assert_eq!(duration_to_ms(before.elapsed()), 12);
    }
    #[test]
    fn test_solve_tsp() {
        use super::solve_tsp;
        use genetic_algorithm_tsp::distance_mat;
        use std::fs;
        // Just run `solve_tsp` for a simple distance matrix.
        // Load in the test matrix.
        let distances = distance_mat::DistanceMat::new(
            fs::read_to_string("tests/test-data/6_cities.txt")
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
        let _ = solve_tsp(&distances, 20, 10, 10, 3);
    }
}
