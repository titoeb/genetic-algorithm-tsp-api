use genetic_algorithm_traits::Individual;
use genetic_algorithm_tsp::distance_mat;
use genetic_algorithm_tsp_api::tsp_solver;
use rocket::serde::json;
use serde::Deserialize;
use serde::Serialize;
use std::time;
#[macro_use]
extern crate rocket;

/// Test whether the API is still alive and can respond.
#[get("/alive")]
fn liveness_probe() -> json::Value {
    json::json!("alive")
}

/// Data that is the input to the `/tsp`-endpoint.
/// Mainly I need this because I cannot implement `Serialize`  or
/// `Deserialize` for the foreign struct `DistanceMat`.
#[derive(Serialize, Deserialize)]
struct SolveTspData {
    distances: Vec<Vec<f64>>,
    n_generations: usize,
}
/// Return type for the `/tsp`-enpoint.
#[derive(Serialize, Deserialize)]
struct RouteWithFitness {
    route: Vec<usize>,
    fitness: f64,
}

/// Main enpoint of the API that takes in a distance matrix and
/// returns the optimal routes.
#[post("/tsp", format = "json", data = "<input_parameters>")]
fn solve_tsp(input_parameters: json::Json<SolveTspData>) -> json::Value {
    let input_parameters: SolveTspData = input_parameters.into_inner();
    // Load in the test matrix.
    let distances = distance_mat::DistanceMat::new(input_parameters.distances);
    // log distance matrix provided.
    println!("{:?}", distances);
    // Get a solution
    let before = time::Instant::now();
    let best_invdividuals =
        tsp_solver::solve_tsp(&distances, input_parameters.n_generations, 30, 10, 3);

    // Log duration.
    let duration = tsp_solver::duration_to_ms(before.elapsed());
    println!("Computation took {}", duration);
    let best_individuals_with_fitness = best_invdividuals
        .iter()
        .map(|individual| RouteWithFitness {
            route: individual.indexes.clone(),
            fitness: -individual.fitness(&distances),
        })
        .collect::<Vec<RouteWithFitness>>();
    json::json!(best_individuals_with_fitness)
}
/// If an enpoint cannot be found, return "Not found!"
#[catch(404)]
fn not_found() -> json::Value {
    json::json!("Not found!")
}

/// If an internal server error happens, return "Your computation could not be done."
#[catch(500)]
fn failed_computation() -> json::Value {
    json::json!("Your computation could not be done.")
}

/// Build Rocket API.
#[launch]
fn rocket() -> _ {
    rocket::build()
        .mount("/", routes![liveness_probe, solve_tsp])
        .register("/", catchers![not_found, failed_computation])
}

#[cfg(test)]
mod test {
    use super::rocket;
    use super::*;
    use rocket::http;
    use rocket::local::blocking;
    use serde_json;

    #[test]
    fn test_not_found() {
        // Test that for an unkown route, "Not found" is returned
        let client = blocking::Client::tracked(rocket()).unwrap();
        let response = client.get("/does/not/exist").dispatch();
        assert_eq!(response.status(), http::Status { code: 404 });
        assert_eq!(response.content_type(), Some(http::ContentType::JSON));
        assert_eq!(
            response.into_string().unwrap(),
            String::from(r##""Not found!""##)
        );
    }
    #[test]
    fn test_liveness() {
        // Test the liveness probe.
        let client = blocking::Client::tracked(rocket()).unwrap();
        let response = client.get("/alive").dispatch();
        assert_eq!(response.status(), http::Status::Ok);
        assert_eq!(response.content_type(), Some(http::ContentType::JSON));
        assert_eq!(
            response.into_string().unwrap(),
            String::from(r##""alive""##)
        );
    }
    #[test]
    fn test_tsp() {
        // Check that the enpoint works with a realistic request.
        // I make the following assumptions on the reponse:
        // - 200 status code
        // - Return type is json
        // - Return value can be deserialized into Vec<RouteWithFitness>
        // - There are three solutions returned.
        let client = blocking::Client::tracked(rocket()).unwrap();
        let response = client
            .post("/tsp")
            .header(http::ContentType::JSON)
            .body(
                r##"{
                "distances": [
                    [0,64,378,519,434,200],
                    [64,0,318,455,375,164],
                    [378,318,0,170,265,344],
                    [519,455,170,0,223,428],
                    [434,375,265,223,0,273],
                    [200,164,344,428,273,0]],
                "n_generations":  10000
                }"##,
            )
            .dispatch();

        assert_eq!(response.status(), http::Status::Ok);
        assert_eq!(response.content_type(), Some(http::ContentType::JSON));
        let returned_routes: Vec<RouteWithFitness> =
            serde_json::from_str(&response.into_string().unwrap()).unwrap();
        assert_eq!(returned_routes.len(), 3);
    }
}
