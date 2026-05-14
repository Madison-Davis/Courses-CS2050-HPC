// run_serial.cpp


// ++++++++ IMPORTS ++++++++ //
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include "Config.h"


// ++++++++ VARIABLES ++++++++ //
#ifndef TEST_MODE
std::string output_positions = "serial_positions.csv";
std::string output_timing = "serial_timing.csv";
#endif


// ++++++++ FUNCTIONS ++++++++ //
void compute_acceleration_serial(std::vector<Body>& bodies) {
    // Compute acceleration due to planet + other bodies

    for (size_t i = 0; i < bodies.size(); ++i) {
        if (!bodies[i].alive) continue;
        // Compute acceleration of central planet at origin
        Vec3 a = {0,0,0};
        Vec3 r = bodies[i].pos;
        double dist = r.norm();
        if (dist > 0) {
            a = a + r * (-Config::G * Config::planet_mass / (dist*dist*dist));
        }
        // Compute acceleration of other bodies
        for (size_t j = 0; j < bodies.size(); ++j) {
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

void check_collisions_serial(std::vector<Body>& bodies) {
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

void integrate_serial(std::vector<Body>& bodies, double dt) {
    // Velocity Verlet integration

    // Update position
    for (auto& b : bodies) {
        if (!b.alive) continue;
        b.pos = b.pos + b.vel * dt + b.acc * (0.5 * dt * dt);
    }
    // Save old acceleration before overwriting
    std::vector<Vec3> acc_old;
    acc_old.reserve(bodies.size());
    for (auto& b : bodies) {
        acc_old.push_back(b.acc);
    }
    // Compute new acceleration and update velocity
    compute_acceleration_serial(bodies);
    for (size_t i = 0; i < bodies.size(); ++i) {
        if (!bodies[i].alive) continue;
        bodies[i].vel = bodies[i].vel + (acc_old[i] + bodies[i].acc) * (0.5 * dt);
    }
}


// ++++++++ MAIN FUNCTION ++++++++ //
#ifndef TEST_MODE
int main() {
    std::vector<Body> bodies;

    // Step 0: Construct a random number generator
    std::mt19937 rng(42); // Fix seed for reproducibility
    std::uniform_real_distribution<double> dist_radius(Config::MIN_ORBIT_RADIUS, Config::MAX_ORBIT_RADIUS);
    std::uniform_real_distribution<double> dist_angle(0.0, 2*M_PI);
    std::uniform_real_distribution<double> dist_inclination(-0.1, 0.1); // small inclination

    // Step 1: Create all bodies
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

    // Step 2: Compute initial acceleration of bodies
    compute_acceleration_serial(bodies);

    // Step 3: Define Output Files
        // Define output file for positions
        // Define output file for timing
    std::ofstream outfile(output_positions);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file!\n";
        return 1;
    }
    outfile << "time,body,x,y,z\n";

    std::ofstream timefile(output_timing);
    if (!timefile.is_open()) {
        std::cerr << "Error opening timing file!\n";
        return 1;
    }
    timefile << "step,time_sec\n";

    // Step 4: Define steps and time variables for simulation
    double dt = 1.0;                // timestep in seconds
    double total_step_time = 0.0;   // sum of per-step times
    int steps = Config::NUM_STEPS;
    auto total_start = std::chrono::high_resolution_clock::now();

    // Step 5: Main simulation loop
    for (int step = 0; step < steps; ++step) {
        // Begin timing this one step
        auto start = std::chrono::high_resolution_clock::now();

        // Advance system state by one timestep
        // Detect and resolve collisions
        integrate_serial(bodies, dt);
        check_collisions_serial(bodies);

        // End timing for this step, accumulate total time step, and save time results
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> step_time = end - start;
        total_step_time += step_time.count();
        timefile << step << "," << step_time.count() << "\n";

        // Save positions of all alive bodies
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

    // Step 6: Compute total simulation time
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;
    double average_step_time = total_step_time / steps;
    std::cout << "Total simulation time: " << total_duration.count() << " seconds\n";
    std::cout << "Average time per step: " << average_step_time << " seconds\n";
    timefile << "total," << total_duration.count() << "\n";
    timefile << "average_per_step," << average_step_time << "\n";
    
    return 0;
}
#endif