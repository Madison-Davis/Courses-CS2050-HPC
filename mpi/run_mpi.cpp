// run_mpi.cpp


// ++++++++ IMPORTS ++++++++ //
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include "Config.h"
#include <mpi.h>


// ++++++++ VARIABLES ++++++++ //
#ifndef TEST_MODE
std::string output_positions = "mpi_positions.csv";
std::string output_timing = "mpi_timing.csv";
#endif

// For MPI_Allgatherv, only subset of data!
struct BodyPos { double x, y, z; };                                     // position-only
struct BodyState { Vec3 vel, acc; double mass, radius; bool alive; };   // essentially everything but position


// ++++++++ FUNCTIONS ++++++++ //
void compute_acceleration_mpi(std::vector<Body>& bodies, int rank, int size) {
    // Compute acceleration due to planet + other bodies

    // MPI: each process will compute a subset of the body accelerations
    // Specifically, each will compute a sized 'chunk' determined by N / size
    // The starting body is at rank*chunk and ending body is at start+chunk
    // If there is an uneven split, let the last rank take on all remainder bodies
    size_t N = bodies.size();
    size_t chunk = N / size;
    size_t start = rank * chunk;
    size_t end   = (rank == size - 1) ? N : start + chunk;

    for (size_t i = start; i < end; ++i) {
        if (!bodies[i].alive) continue;
        // Compute acceleration of central planet at origin
        Vec3 a = {0,0,0};
        Vec3 r = bodies[i].pos;
        double dist = r.norm();
        if (dist > 0) {
            a = a + r * (-Config::G * Config::planet_mass / (dist*dist*dist));
        }
        // Compute acceleration of other bodies
        for (size_t j = 0; j < N; ++j) {
            if (i == j || !bodies[j].alive) continue;
            Vec3 dr = bodies[j].pos - bodies[i].pos;
            double d = dr.norm();
            if (d > 0) {
                a = a + dr * (Config::G * bodies[j].mass / (d*d*d));
            }
        }
        bodies[i].acc = a;
    }
}

void check_collisions_mpi(std::vector<Body>& bodies) {
    // Collision detection between planet + other bodies

    for (size_t i = 0; i < bodies.size(); ++i) {
        if (!bodies[i].alive) continue;
        // Collision with planet: if so, destroy body
        if (bodies[i].pos.norm() <= Config::planet_radius) {
            bodies[i].alive = false;
            continue;
        }
        // Collision with other bodies: if so, merge bodies
        for (size_t j = i+1; j < bodies.size(); ++j) {
            if (!bodies[j].alive) continue;
            double d = (bodies[i].pos - bodies[j].pos).norm();
            if (d <= (bodies[i].radius + bodies[j].radius)) {
                double total_mass = bodies[i].mass + bodies[j].mass;
                bodies[i].vel = (bodies[i].vel * bodies[i].mass + bodies[j].vel * bodies[j].mass) / total_mass;
                bodies[i].mass = total_mass;
                bodies[i].radius = std::cbrt(std::pow(bodies[i].radius,3) + std::pow(bodies[j].radius,3)); // volume add
                bodies[j].alive = false;
            }
        }
    }
}

void integrate_mpi(std::vector<Body>& bodies, double dt, int rank, int size,
                   const std::vector<int>& counts, const std::vector<int>& displs,
                   const std::vector<int>& pos_counts, const std::vector<int>& pos_displs,
                   const std::vector<int>& state_counts, const std::vector<int>& state_displs) {
    // Velocity Verlet integration

    // MPI: each process will compute a subset of the body integrations
    // Specifically, each will compute a sized 'chunk' determined by N / size
    // The starting body is at rank*chunk and ending body is at start+chunk
    // If there is an uneven split, let the last rank take on all remainder bodies
    size_t N = bodies.size();
    size_t chunk = N / size;
    size_t start = rank * chunk;
    size_t end   = (rank == size - 1) ? N : start + chunk;

    // Update positions for this specific rank
    for (size_t i = start; i < end; ++i) {
        if (!bodies[i].alive) continue;
        bodies[i].pos = bodies[i].pos + bodies[i].vel * dt + bodies[i].acc * (0.5 * dt * dt);
    }
    // Sync updated positions across all ranks
    // Instead of Allgatherv on full bodies after position update,
    // build a compact pos-only buffer!
    std::vector<BodyPos> pos_buf(N);
    for (size_t i = start; i < end; ++i)
        pos_buf[i] = {bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z};
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                pos_buf.data(), pos_counts.data(), pos_displs.data(),
                MPI_BYTE, MPI_COMM_WORLD);
    // Write back
    for (size_t i = 0; i < N; ++i) {
        bodies[i].pos.x = pos_buf[i].x;
        bodies[i].pos.y = pos_buf[i].y;
        bodies[i].pos.z = pos_buf[i].z;
    }

    // Save old acceleration before overwriting
    std::vector<Vec3> acc_old(N);
    for (size_t i = 0; i < N; ++i) {
        acc_old[i] = bodies[i].acc;
    }

    // Compute new acceleration locally 
    // No need to AllGatherv: why?
    // Each rank for a chunk [start, end) has already computed the correct acc[i]
    // For exactly the same bodies it has
    compute_acceleration_mpi(bodies, rank, size);

    // Update velocity for this specific rank
    for (size_t i = start; i < end; ++i) {
        if (!bodies[i].alive) continue;
        bodies[i].vel = bodies[i].vel + (acc_old[i] + bodies[i].acc) * (0.5 * dt);
    }
    // Sync updated velocities across all ranks
    // Instead of Allgatherv on full bodies after velocity update,
    // build a compact subset-state buffer!
    std::vector<BodyState> state_buf(N);
    for (size_t i = start; i < end; ++i)
        state_buf[i] = {bodies[i].vel, bodies[i].acc, bodies[i].mass, bodies[i].radius, bodies[i].alive};
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                state_buf.data(), state_counts.data(), state_displs.data(),
                MPI_BYTE, MPI_COMM_WORLD);
    // Write back
    for (size_t i = 0; i < N; ++i) {
        bodies[i].vel    = state_buf[i].vel;
        bodies[i].acc    = state_buf[i].acc;
        bodies[i].mass   = state_buf[i].mass;
        bodies[i].radius = state_buf[i].radius;
        bodies[i].alive  = state_buf[i].alive;
    }
}


// ++++++++ MAIN FUNCTION ++++++++ //
#ifndef TEST_MODE
int main() {

    // MPI: initialize
    MPI_Init(nullptr, nullptr);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<Body> bodies;

    // Step 0: Construct a random number generator
    std::mt19937 rng(42); // Fix seed for reproducibility
    std::uniform_real_distribution<double> dist_radius(Config::MIN_ORBIT_RADIUS, Config::MAX_ORBIT_RADIUS);
    std::uniform_real_distribution<double> dist_angle(0.0, 2*M_PI);
    std::uniform_real_distribution<double> dist_inclination(-0.1, 0.1); // small inclination

    // Step 1: Create all bodies
    // To save Bcast command, let all ranks compute simultaneously 
    for (int i = 0; i < Config::NUM_BODIES; ++i) {
        double r = dist_radius(rng);
        double theta = dist_angle(rng);         // Angle in xy-plane
        double phi = dist_inclination(rng);     // Small inclination

        // Set position in 3D spherical coordinates
        Body b;
        b.pos.x = r * std::cos(theta) * std::cos(phi);
        b.pos.y = r * std::sin(theta) * std::cos(phi);
        b.pos.z = r * std::sin(phi);

        // Set circular orbit velocity magnitude
        double v_circ = std::sqrt(Config::G * Config::planet_mass / r);

        // Set velocity perpendicular to position vector (approximate)
        // Use simple rotation in xy-plane and z-component for small inclination
        b.vel.x = -v_circ * std::sin(theta);
        b.vel.y =  v_circ * std::cos(theta);
        b.vel.z = 0;
        b.mass = Config::BODY_MASS;
        b.radius = Config::BODY_RADIUS;

        // Add body to the vector of bodies
        bodies.push_back(b);
    }

    // Precompute counts and displacements for MPI_Allgatherv
        // 1. all data
        // 2. positions only
        // 3. velocity/acc/mass/radius/alive
    std::vector<int> counts(size), displs(size);
    std::vector<int> pos_counts(size), pos_displs(size);
    std::vector<int> state_counts(size), state_displs(size);
    size_t chunk = Config::NUM_BODIES / size;
    for (int r = 0; r < size; ++r) {
        size_t r_start = r * chunk;
        size_t r_end   = (r == size - 1) ? Config::NUM_BODIES : r_start + chunk;
        size_t n       = r_end - r_start;
        counts[r]       = n * sizeof(Body);
        displs[r]       = r_start * sizeof(Body);
        pos_counts[r]   = n * sizeof(BodyPos);
        pos_displs[r]   = r_start * sizeof(BodyPos);
        state_counts[r] = n * sizeof(BodyState);
        state_displs[r] = r_start * sizeof(BodyState);
    }
            
    // Step 2: Compute initial acceleration of bodies and gather results to all
    compute_acceleration_mpi(bodies, rank, size);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
               bodies.data(), counts.data(), displs.data(),
               MPI_BYTE, MPI_COMM_WORLD);

    // Step 3: Define Output Files
        // Define output file for positions
        // Define output file for timing
    std::ofstream outfile, timefile;
    if (rank == 0) {
        outfile.open(output_positions);
        if (!outfile.is_open()) { std::cerr << "Error opening file!\n"; return 1; }
        outfile << "time,body,x,y,z\n";
        timefile.open(output_timing);
        if (!timefile.is_open()) { std::cerr << "Error opening timing file!\n"; return 1; }
        timefile << "step,time_sec\n";
    }

    // Step 4: Define steps and time variables for simulation
    double dt = 1.0;                // timestep in seconds
    double total_step_time = 0.0;   // sum of per-step times
    int steps = Config::NUM_STEPS;
    auto total_start = std::chrono::high_resolution_clock::now();

    // Step 5: Main simulation loop
    for (int step = 0; step < steps; ++step) {
        // Begin timing this one step
        // Only rank 0 records timing, but ranks can exit integrate_mpi's
        // Allgatherv at slightly different wall-clock times
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

        // Advance system state by one timestep
        // Integrate mpi per rank, ensuring consistency internally via Allgatherv
        integrate_mpi(bodies, dt, rank, size, counts, displs, pos_counts, pos_displs, state_counts, state_displs);
        // Detect collisions just on rank 0 to ensure identical state, Bcast later
        if (rank == 0) {
            check_collisions_mpi(bodies);
        }
        int total_bytes = displs[size-1] + counts[size-1];
        MPI_Bcast(bodies.data(), total_bytes, MPI_BYTE, 0, MPI_COMM_WORLD);

        // End timing for this step, accumulate total time step, and save time results
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> step_time = end - start;
        total_step_time += step_time.count();
        if (rank == 0) {
            timefile << step << "," << step_time.count() << "\n";
        }

        // Save positions of all alive bodies
        if (rank == 0) {
            for (size_t i = 0; i < bodies.size(); ++i) {
                    if (!bodies[i].alive) continue;
                    outfile << step*dt << "," << i << "," 
                            << bodies[i].pos.x << "," 
                            << bodies[i].pos.y << "," 
                            << bodies[i].pos.z << "\n";
                }
                // Occasionally print state to console (every 100 steps)
                if (step % 100 == 0) {
                    std::cout << "Step " << step << ":\n";
                    for (size_t i = 0; i < bodies.size(); ++i) {
                        if (!bodies[i].alive) continue;
                        std::cout << "Body " << i << ": (" << bodies[i].pos.x << "," 
                                << bodies[i].pos.y << "," << bodies[i].pos.z << ")\n";
                    }
                }
        }
    }

    // Step 6: Compute total simulation time
    if (rank == 0) {
        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_duration = total_end - total_start;
        double average_step_time = total_step_time / steps;
        std::cout << "Total simulation time: " << total_duration.count() << " seconds\n";
        std::cout << "Average time per step: " << average_step_time << " seconds\n";
        timefile << "total," << total_duration.count() << "\n";
        timefile << "average_per_step," << average_step_time << "\n";
    }

    // MPI: finalize
    MPI_Finalize();

    return 0;
}
#endif