// run_kokkos.cpp


// ++++++++ IMPORTS ++++++++ //
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include "Config.h"
#include <Kokkos_Core.hpp>


// ++++++++ VARIABLES ++++++++ //
#ifndef TEST_MODE
std::string output_positions = "kokkos_positions.csv";
std::string output_timing = "kokkos_timing.csv";
#endif


// ++++++++ FUNCTIONS ++++++++ //
void compute_acceleration_kokkos(
    int N,
    Kokkos::View<double*> pos_x, Kokkos::View<double*> pos_y, Kokkos::View<double*> pos_z,
    Kokkos::View<double*> acc_x, Kokkos::View<double*> acc_y, Kokkos::View<double*> acc_z,
    Kokkos::View<double*> mass,  Kokkos::View<int*> alive
) {
    // Compute acceleration due to planet + other bodies
    
    Kokkos::parallel_for("compute_accel", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(int i) {
            if (!alive(i)) return;
            // Compute acceleration of central planet at origin
            double xi = pos_x(i), yi = pos_y(i), zi = pos_z(i);
            double axi = 0, ayi = 0, azi = 0;
            double dist = sqrt(xi*xi + yi*yi + zi*zi);
            if (dist > 0) {
                double f = -Config::G * Config::planet_mass / (dist*dist*dist);
                axi += xi * f; ayi += yi * f; azi += zi * f;
            }
            // Compute acceleration of other bodies
            for (int j = 0; j < N; ++j) {
                if (j == i || !alive(j)) continue;
                double dx = pos_x(j) - xi, dy = pos_y(j) - yi, dz = pos_z(j) - zi;
                double d2 = dx*dx + dy*dy + dz*dz;
                double inv = Kokkos::rsqrt(d2 + 1e-9);
                double f = Config::G * mass(j) * inv * inv * inv;
                axi += dx * f; ayi += dy * f; azi += dz * f;
            }
            acc_x(i) = axi; acc_y(i) = ayi; acc_z(i) = azi;
        }
    );
}

void check_collisions_kokkos(
    int N,
    Kokkos::View<double*> pos_x, Kokkos::View<double*> pos_y, Kokkos::View<double*> pos_z,
    Kokkos::View<double*> vel_x, Kokkos::View<double*> vel_y, Kokkos::View<double*> vel_z,
    Kokkos::View<double*> mass,  Kokkos::View<double*> radius,
    Kokkos::View<int*> alive
) {
    // Collision detection between planet + other bodies

    // Collision with planet: if so, destroy body
    Kokkos::parallel_for("collisions", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(int i) {
            if (!alive(i)) return;
            double d = sqrt(pos_x(i)*pos_x(i) + pos_y(i)*pos_y(i) + pos_z(i)*pos_z(i));
            if (d <= Config::planet_radius) alive(i) = 0;
        }
    );
    // Collision with other bodies: if so, merge bodie
    // Compute on host, mirrors serial.cpp version
    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pos_x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pos_y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pos_z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vel_x);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vel_y);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vel_z);
    auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mass);
    auto h_r  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), radius);
    auto h_a  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), alive);
    for (int i = 0; i < N; ++i) {
        if (!h_a(i)) continue;
        for (int j = i + 1; j < N; ++j) {
            if (!h_a(j)) continue;
            double dx = h_x(j)-h_x(i), dy = h_y(j)-h_y(i), dz = h_z(j)-h_z(i);
            if (sqrt(dx*dx + dy*dy + dz*dz) <= (h_r(i) + h_r(j))) {
                double total_mass = h_m(i) + h_m(j);
                // Similar to serial.cpp, update velocity, mass, radius, and if alive
                h_vx(i) = (h_vx(i)*h_m(i) + h_vx(j)*h_m(j)) / total_mass;
                h_vy(i) = (h_vy(i)*h_m(i) + h_vy(j)*h_m(j)) / total_mass;
                h_vz(i) = (h_vz(i)*h_m(i) + h_vz(j)*h_m(j)) / total_mass;
                h_m(i)  = total_mass;
                h_r(i)  = cbrt(h_r(i)*h_r(i)*h_r(i) + h_r(j)*h_r(j)*h_r(j));
                h_a(j)  = 0;
            }
        }
    }
    Kokkos::deep_copy(vel_x, h_vx); Kokkos::deep_copy(vel_y, h_vy); Kokkos::deep_copy(vel_z, h_vz);
    Kokkos::deep_copy(mass,  h_m);  Kokkos::deep_copy(radius, h_r); Kokkos::deep_copy(alive, h_a);
}

void integrate_kokkos(
    int N, double dt,
    Kokkos::View<double*> pos_x, Kokkos::View<double*> pos_y, Kokkos::View<double*> pos_z,
    Kokkos::View<double*> vel_x, Kokkos::View<double*> vel_y, Kokkos::View<double*> vel_z,
    Kokkos::View<double*> acc_x, Kokkos::View<double*> acc_y, Kokkos::View<double*> acc_z,
    Kokkos::View<double*> mass,  Kokkos::View<int*> alive
) {
    // Velocity Verlet integration

    // Update position
    Kokkos::parallel_for("integrate_pos", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(int i) {
            if (!alive(i)) return;
            pos_x(i) += vel_x(i)*dt + 0.5*acc_x(i)*dt*dt;
            pos_y(i) += vel_y(i)*dt + 0.5*acc_y(i)*dt*dt;
            pos_z(i) += vel_z(i)*dt + 0.5*acc_z(i)*dt*dt;
        }
    );
    // Save old acceleration before overwriting
    Kokkos::View<double*> acc_old_x("acc_old_x", N);
    Kokkos::View<double*> acc_old_y("acc_old_y", N);
    Kokkos::View<double*> acc_old_z("acc_old_z", N);
    Kokkos::deep_copy(acc_old_x, acc_x);
    Kokkos::deep_copy(acc_old_y, acc_y);
    Kokkos::deep_copy(acc_old_z, acc_z);
    // Compute new acceleration and update velocity
    compute_acceleration_kokkos(N, pos_x, pos_y, pos_z, acc_x, acc_y, acc_z, mass, alive);
    Kokkos::parallel_for("integrate_vel", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(int i) {
            if (!alive(i)) return;
            vel_x(i) += 0.5*(acc_old_x(i) + acc_x(i))*dt;
            vel_y(i) += 0.5*(acc_old_y(i) + acc_y(i))*dt;
            vel_z(i) += 0.5*(acc_old_z(i) + acc_z(i))*dt;
        }
    );
}


// ++++++++ MAIN FUNCTION ++++++++ //
#ifndef TEST_MODE
int main(int argc, char* argv[]) {

    // Kokkos: initialize
    Kokkos::initialize(argc, argv);
    {
        std::cout << "Kokkos threads: "
          << Kokkos::DefaultExecutionSpace().concurrency()
          << std::endl;
          
        int N = Config::NUM_BODIES;

        Kokkos::View<double*> pos_x("x", N), pos_y("y", N), pos_z("z", N);
        Kokkos::View<double*> vel_x("vx", N), vel_y("vy", N), vel_z("vz", N);
        Kokkos::View<double*> acc_x("ax", N), acc_y("ay", N), acc_z("az", N);
        Kokkos::View<double*> mass("m", N);
        Kokkos::View<double*> radius("r", N);
        Kokkos::View<int*>    alive("alive", N);

        auto h_x  = Kokkos::create_mirror_view(pos_x);
        auto h_y  = Kokkos::create_mirror_view(pos_y);
        auto h_z  = Kokkos::create_mirror_view(pos_z);
        auto h_vx = Kokkos::create_mirror_view(vel_x);
        auto h_vy = Kokkos::create_mirror_view(vel_y);
        auto h_vz = Kokkos::create_mirror_view(vel_z);
        auto h_m  = Kokkos::create_mirror_view(mass);
        auto h_r  = Kokkos::create_mirror_view(radius);
        auto h_a  = Kokkos::create_mirror_view(alive);

        // Step 0: Construct a random number generator
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> rdist(Config::MIN_ORBIT_RADIUS, Config::MAX_ORBIT_RADIUS);
        std::uniform_real_distribution<double> adist(0, 2*M_PI);
        std::uniform_real_distribution<double> idist(-0.1, 0.1);

        // Step 1: Create all bodies
        for (int i = 0; i < N; ++i) {
            double r = rdist(rng), t = adist(rng), p = idist(rng);
            h_x(i) = r*cos(t)*cos(p);
            h_y(i) = r*sin(t)*cos(p);
            h_z(i) = r*sin(p);
            double v = sqrt(Config::G * Config::planet_mass / r);
            h_vx(i) = -v*sin(t);
            h_vy(i) =  v*cos(t);
            h_vz(i) = 0;
            h_m(i) = Config::BODY_MASS;
            h_r(i) = Config::BODY_RADIUS;
            h_a(i) = 1;
        }

        Kokkos::deep_copy(pos_x, h_x); Kokkos::deep_copy(pos_y, h_y); Kokkos::deep_copy(pos_z, h_z);
        Kokkos::deep_copy(vel_x, h_vx); Kokkos::deep_copy(vel_y, h_vy); Kokkos::deep_copy(vel_z, h_vz);
        Kokkos::deep_copy(mass, h_m); Kokkos::deep_copy(radius, h_r); Kokkos::deep_copy(alive, h_a);

        // Step 2: Define Output Files
            // Define output file for positions
            // Define output file for timing
        std::ofstream outfile(output_positions);
        if (!outfile.is_open()) {
            std::cerr << "Error opening positions file!\n";
            return 1; 
        }
        outfile << "time,body,x,y,z\n";

        std::ofstream timefile(output_timing);
        if (!timefile.is_open()) { 
            std::cerr << "Error opening timing file!\n";
            return 1; 
        }
        timefile << "step,time_sec\n";

        // Step 3: Define steps and time variables for simulation
        double dt = 1.0;                // timestep in seconds
        double total_step_time = 0.0;   // sum of per-step times
        int steps = Config::NUM_STEPS;
        auto total_start = std::chrono::high_resolution_clock::now();

        // Step 4: Main simulation loop
        for (int step = 0; step < steps; ++step) {
            // Begin timing this one step
            auto start = std::chrono::high_resolution_clock::now();

            // Advance system state by one timestep
            // Detect and resolve collisions
            integrate_kokkos(N, dt,
                pos_x, pos_y, pos_z,
                vel_x, vel_y, vel_z,
                acc_x, acc_y, acc_z,
                mass, alive);
            check_collisions_kokkos(N, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, radius, alive);

            Kokkos::fence();

            // End timing for this step, accumulate total time step, and save time results
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> step_time = end - start;
            total_step_time += step_time.count();
            timefile << step << "," << step_time.count() << "\n";

            // Save positions of all alive bodies, this time every 10 steps
            if (step % 10 == 0) {
                Kokkos::deep_copy(h_x, pos_x);
                Kokkos::deep_copy(h_y, pos_y);
                Kokkos::deep_copy(h_z, pos_z);
                Kokkos::deep_copy(h_a, alive);
                for (int i = 0; i < N; ++i) {
                    if (!h_a(i)) continue;
                    outfile << step*dt << "," << i << ","
                            << h_x(i) << "," << h_y(i) << "," << h_z(i) << "\n";
                }
            }
            // Occasionally print state to console (every 100 steps)
            if (step % 100 == 0) {
                std::cout << "Step " << step << " complete\n";
            }
        }

        // Step 5: Compute total simulation time
        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_duration = total_end - total_start;
        double average_step_time = total_step_time / steps;
        std::cout << "Total simulation time: " << total_duration.count() << " seconds\n";
        std::cout << "Average time per step: " << average_step_time << " seconds\n";
        timefile << "total," << total_duration.count() << "\n";
        timefile << "average_per_step," << average_step_time << "\n";
    }

    // Kokkos: finalize
    Kokkos::finalize();
    return 0;
}
#endif