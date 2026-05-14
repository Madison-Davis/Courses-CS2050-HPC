
# CS 2050 Final Project
# Algorithm Overview and Main Methods
This project simulates an N-body spatial orbital collision environment.  There is one main planetary body and N-bodies orbiting it.  Here is a visualization video, generated from this project's serial position data: <br>
https://github.com/user-attachments/assets/f4dc0c54-b97b-496a-b606-7aca4eabb7cd
<img width="1623" height="1003" alt="Screenshot 2026-05-14 at 9 51 22 AM" src="https://github.com/user-attachments/assets/64c46a53-6dea-4204-8f71-a00817ef00a3" />

The algorithmic simulation involves three main methods:
- `compute_acceleration`: for each body (both for the planet and the N bodies), update its new acceleration using the formula `a' = a + r * F` where `a` stands for acceleration, `r` for position, and `F` for Newton's Law of Universal Gravitation.
- `check_collisions`: for each of the N-bodies, I check for collisions.  First, if a body collides with the surface of the planet (by comparing the body's Euclidian length of its position with the planet's radius), we destroy the incoming body.  Second, if two bodies i and j collide with one another, they merge under a perfectly inelastic model: the new mass is m(i) + m(j); radius is v(i)^1/3 + v(j)^1/3, where v is volume; and velocity is m(i)v(i) * m(j)v(j).
- `integrate`: for each of the N-bodies, I update its position and velocity values using Velocity Verlet integration equations.




# Parallelization Strategies
`run_serial.cpp` implements the three aforementioned methods without parallelization.
- There is also a Config.h file in the "additional" folder that contains definitions for a Vector3 struct (x/y/z and operators), Body struct (position, velocity, acceeration, mass, radius, and ifalive), BodyGPU struct (identical to Body, but written out as a struct-of-array format versus array-of-struct format, the former of whch is more comaptible with GPU computation), and global variables (num bodies, body radius, gravitational constants, and more).  All 5 parts use this file.  <br>

`run_openmp.cpp` adds the following changes: <br>
- `compute_acceleration_openmp(vector<Body>)`: here, I use #pragma omp parallel for schedule(dynamic) over the loop of bodies, computing their updated acceleration.  I can do this because each body's acceleration is solely dependent on (1) its distance from each other body, which is fixed at the time of computation; and (2) its and surrounding bodies' respective masses (collisions are dealt with after updating acceleration).  I define an accumulation vector Vec3 a = {0,0,0} specific to each body that accumulates all interactions from nearby paired-objects, so this results in one vector per thread.
- `integrate_bodies_openmp(vector<Body>)`: here, I parallelize the loop computing new positions and the loop for new velociites using #pragma omp parallel for, respectively. <br>

`run_mpi.cpp` adds the following changes: <br>
- `compute_acceleration(vector<Body>, rank, size)`: each process will compute a subset (N/size) of the bodies' accelerations or integrations, respectively.  N is the total number of bodies and size is the number of processes.  Because N may not evenly divide by size, the last rank is given the remaining excess bodies.
- `integrate_mpi(bodies, dt, rank, size, counts, displs, pos_counts, pos_displs, state_counts, state_displs)`: same strategy as above.
 
Written below is a detailed description and justification analysis of the MPI communication strategy (performance analysis is discussed in the next section):
- `compute_acceleration` work is split per rank, and the function has no internal MPI calls.  This is intentional because compute_acceleration will be later used in the integrate function call and, as I describe later, does not need to be synced across all ranks for it to proceed in the integrate function.
- `check_collisions` work is done serially because it destroys/merges some bodies in the process, and so order dependence can determine what bodies get destroyed/merged.
- `integrate` work is split per rank.  First, each rank updates its bodies' positions.  Then, I call a partial (position-only) MPI_Allgatherv to sync all updated positions.  Second, I compute the new accelerations via compute_acceleration.  Each rank at this point has only its subset of bodies' updated accelerations.  Finally, each rank updates its own bodies' velocities.  Each body's velocity only depends on its own acceleration, and this knowledge is already located in the same rank.  I call a partial (i.e on all fields except positions) MPI_Allgatherv to sync the remaining fields.  I do Allgather 'v' to allow for variable chunk size, and it is a 'partial' Allgatherv so I do not have to send information that is already updated on either end.
- `main()` does the following:
   - First, I initialize an MPI region (MPI_Init and MPI_Finalize).
   - Second, each rank creates all the bodies.  This safe to do because I use the same generation seed and it avoids a rank 0 computation plus MPI_BCast operation, which can take time to complete.  Do note it is entirely plausible to also do this work just on rank 0, then MPI_BCast.
   - Third, I compute bodies' initial accelerations via compute_acceleration, then use MPI_Allgatherv to sync this state across all ranks. 
   - Fouth, the main timestep proceeds as follows:
       - Before timing each step, I do MPI_Barrier(MPI_COMM_WORLD) to ensure all ranks are synced before running this step.
       - Every rank integrates its assigned chunk of bodies, which as mentioend previously, uses MPI_AllGatherv internally to ensure all fields are synced by the end of the call.
       - I then run check_collisions_mpi on rank 0 only (the shared body states are edited when computing, and so doing this computation on multiple ranks could cause order dependent output).  Then, I call MPI_Bcast to sync the results.
- Finally, output files are written only by rank 0 to avoid duplicate or garbled output.

`run_cuda.cpp` adds the following changes:
- `compute_acceleration_cuda(BodyGPU, N)`: this function is now a wrapper function that is called on the host (CPU) to set up and run a GPU kernel, compute_acceleration_kernel.
    - The wrapper function takes in a BodyGPU (struct of arrays) compared to a Body struct.  This is because the struct of array layout offers contiguous access on a GPU, who use 32 threads in a warp-like fashion to issue the same instruction on different data.  In a struct of arrays, pos_x[0] and pos_x[1] will be adjacent (used for threads i/i+1) because they exist side-by-side in the same array pos_x, whereas in an array of structs, one would have to pass over other values to get to the next body's pos_x value.
    - The wrapper function first determines the number of blocks and threads to send to the kernel, defined by this formula: blocks = (N + THREADS - 1) / THREADS.  N refers to the number of bodies, and THREADS the number of THREADS (statically defined as 256).  The THREADS - 1 in the numerator forces an arithmetic round-up so that all bodies are computed on, akin to a ceil.
- `check_collisions_cuda(bodyGPU, N)`: same wrapper setup as compute_acceleration_cuda.
- `integrate_cuda(bodyGPU, N, dt)`: same wrapper setup as compute_acceleration_cuda.
- The kernel functions that the cuda functions wrap around perform the same underlying mathematics as the original functions.  One additional step for all is that they first compute which body each thread handles using the formula int i = blockIdx.x * blockDim.x + threadIdx.x.
- I also implement additional memory functions that help create/move data to/from the device.  These include allocBodyGPU(N), which creates N bodies; copyBodiesToGPU; copyBodiesFromGPU; copyPositionsFromGPU (just copy the position values); and freeBodyGPU, which is specifically used for cuda's test file.
- `main()`: I initialize all bodies on the CPU (host).  I then allocate the bodies to GPU memory using copyBodiesToGPU.  Once this is done, I proceed as normal compared to the serial version, with one caveat: anytime one of the three main cuda-version functions is called, I call cudaDeviceSynchronize after to double-triple ensure the computation completes before I use it.


`run_kokkos.cpp` adds the following changes:
- `compute_acceleration_kokkos(N, Kokkos::View<double*> arrays ...)`: to parallelize the acceleration computation for each of the N bodies, I use Kokkos::parallel_for("compute_accel", Kokkos::RangePolicy<>(0, N) and create a KOKKOS_LAMBDA function that the Range operates over the bodies.  Recall that Kokkos::View is the primary data structure in Kokkos, a multidimensional-aware array structure.  
- `check_collisions_kokkos(N, Kokkos::View<double*> arrays ...)`: similar to the acceleration function, to parallelize the collision-detection computation between each body and the planet, I use Kokkos::parallel_for with a KOKKOS_LAMBDA.  For the collisions between each body and other bodies, it was left serialized because the collision handling involves modifying shared state (masses, velocities, radius, alive) and merging bodies, which can result in output order dependence if naively parallelized.
- `integrate_kokkos(N, dt, Kokkos::View<double*> arrays ...)`: similar to the acceleration function, to parallelize the position and velocity integration respectively, I use Kokkos::parallel_for with a KOKKOS_LAMBDA.  I also call the new compute_acceleration_kokkos to perform the updated acceleration before the velocity loop.  Finally, I do a Kokkos::deep_copy of the old acceleration x, y, and z values to use for the velocity loop.
- `main()`: in the main function, I initialize a region for Kokkos using Kokkos::initialize and Kokkos::finalize.
    - For the bodies, I instantiate a Kokkos::View<double*> array for each field possibility (pos_x, pos_y, pos_z, vel_x, etc.).  I also create a Kokkos::create_mirror_view so the host can access these fields.  Note that this struct of array format is similar to run_cuda and differs from serial's std::vector<Body> array of structs format.  This is especially useful if we decide to compile Kokkos with CUDA enabled.
    - I use Kokkos::fence() before the end of each step to give a proper sync barrier.
 
Written below details my experience with Kokkos (performance is discussed in the next section):
- Kokkos is a higher-abstraction compared to CUDA or OpenMP.  Personally, I found Kokkos generally easy to implement, after having to deal with the data layout change.  One part that was incredibly tricky was properly downloading Kokkos' build dependencies and getting it to compile scripts written with Kokkos features.  I had to ensure I had the proper libraries, correct versions of compilers, and right flag extensions.
- Please note that there are two submit.slurm files for Kokkos.  The default is submit.slurm and enables OpenMP, submit_cuda.slurm enables CUDA.  CUDA requires a GPU versus OpenMP I wanted to run on a CPU, which is why they have different #SBATCH setups.


# Key Performance Scaling Results
## Part 1: Timing
I present the timing results (in seconds) across all implementations.  Results are reported to the hundredths place.  All are tested with N=5000 bodies.  All files were run several times over several days at different times of the day to ensure confidence in the numbers.  Note the following assumptions:
- OpenMP checks across {1, 2, 4, 8, 16} threads.
- MPI checks across {1, 2, 4, 8, 16} processes and 8 (> 1) nodes.

| Part        | Total Time (s) 1 (thr/proc) | 2 (thr/proc) |4 (thr/proc) | 8 (thr/proc) | 16 (thr/proc)  |
|------------------|------------------| ----- | ----- | ----- | ----- |
| Serial               | 6.27  | | | | |
| CUDA                   | 1.70 | | | | |
| Kokkos CUDA          | 1.80 | | | | |
| Kokkos OpenMP        | 2.57 | 2.51 | 2.48 | 2.46 | 2.46 |
| OpenMP               | 6.42 | 4.25 | 3.13 | 2.52 | 2.48 |
| MPI                   | 7.04 | 4.85 | 3.72 | 3.62 | 3.26 |


My analysis is as follows:
- Of the CPU implementations, Kokkos OpenMP has the best result.
- Of the GPU implementations, CUDA has the best result (and best overall).
- OpenMP decreases in time as we increase the number of threads.  This shows the benefits of parallelizing across many threads. However, the gains in speed decrease as we increase the number of threads to 16.  This is because, for a given problem setup, the serial portion stays fixed, causing diminishing returns (Amdahl's Law).  OpenMP's time at 1 thread is higher than the serial version, and this is due to the overhead OpenMP puts on the system without any parallelization.
- MPI decreases in time as we increase the number of processors.  MPI for the same number of threads/processes is always worse than OpenMP, and this speaks to the fact that processes have to communicate explicitly (DMP) over nodes (farther) versus OpenMP's easier construction of threads.
- Kokkos' CUDA was slower than CUDA.  This gives evidence for how Kokkos offers portability in compilation at the cost of lower performance.
- Kokkos' OpenMP produced faster results than OpenMP.  This is likely because Kokkos' Views were written with its data structured as structs of arrays, allowing faster contiguous memory accesses versus arrays of structs.  Moreover, in Kokkos, I had used optimized functions such as Kokkos::rsqrt() versus the generic sqrt().  One caveat is that Kokkos' OpenMP saw very little speedup even when I added more threads (it even appeared to reach a stopping-point in gains towards the 8-16 thread mark, at which point OpenMP had essentially caught up).  This is likely due to a combination of Amdahl’s law from the serialized check-collisions section and overhead from frequent deep copy and host-mirror operations required for data movement in the collision step.  This goes to show the importance of an algorithm's setup on performance.

## Part 2: OpenMP + MPI Strong Scaling and Weak Scaling
The OpenMP and MPI strong and weak scaling results are as follows.  Note the following assumptions:
- Strong Scaling for OpenMP/MPI was defined with N=5000 and distributed across {1, 2, 4, 8, 16} threads/processes, respectively.  MPI worked on 8 (> 1) nodes.
- Weak Scaling for OpenMP/MPI was defined with X000:X bodies to threads/processes, respectively.  For example, (2000 bodies, 2 threads) and (4000 bodies, 4 threads) for OpenMP.

| Strong Scaling: OpenMP |
|------------------|
| <img width="804" height="339" alt="Screenshot 2026-05-14 at 9 53 09 AM" src="https://github.com/user-attachments/assets/92af6ce0-9293-4531-8d94-802bf3e5130a" /> |

| Strong Scaling: MPI |
|------------------|
| <img width="889" height="376" alt="Screenshot 2026-05-14 at 9 53 16 AM" src="https://github.com/user-attachments/assets/83ac207f-9479-477c-907e-581e12e06ae5" /> | <br>


| Weak Scaling: OpenMP | Weak Scaling: MPI |
|------------------|------------------|
| <img width="472" height="346" alt="Screenshot 2026-05-14 at 9 53 22 AM" src="https://github.com/user-attachments/assets/49ee1305-f934-4402-8e3a-ef3ff9bf989b" /> | <img width="490" height="352" alt="Screenshot 2026-05-14 at 9 53 28 AM" src="https://github.com/user-attachments/assets/efb2a6cc-6f89-4c44-b4aa-b0e59c357a42" /> | <br>


My analysis is as follows:
- For strong scaling, both OpenMP and MPI do not follow the ideal y=x curve and are generally concave down, peaking around 16 threads/processes for each.  At every tested thread/process, the speedup and the efficiency of strong scaling is the same or weaker for MPI versus OpenMP. This is because of their fundamentally different memory models. Recall strong scaling keeps the same problem size but increases more processes. OpenMP threads use shared memory, which doesn't require explicit communication between threads. In contrast, MPI processes have distributed memory spaces. For compute acceleration, where every process needs position data for all N bodies to compute accelerations, one has to pay inter-process communication across nodes (farther) at each step.
- For weak scaling, both OpenMP and MPI do not follow the ideal y=1 curve and feature a decreasing form.  For every tested thread/process, OpenMP is on par or worse compared to MPI.  This may due to weak scaling's implications on cache pressure.  For OpenMP, all threads share the same memory bus on a node, so as thread count grows, memory bandwidth becomes a bottleneck.  In contrast, MPI processes on distributed nodes have access to independent memory buses. 

## Part 3: CUDA Scaling
The final analysis relates to CUDA Scaling.  The premise is this: some GPU workloads are linear (more work won't worsen runtime more than linearly) while others are not.  Is my workload of N=5000 bodies on the part of the scale where it can properly fit in GPU memory or not?  I conduct a study with the following N values: (128 256 512 1024 2048 4096 5000 8192 16384 32768 65536 131072).  The results are shown below, with regressed lines per region.  The regressions are done as power law equations, time vs number of bodies (t=aN^b), where b indicates the speedup amount.  To give some intuition:
- b ~= 0 is flat.  Here the GPU is underutilised, where adding extra bodies would cost almost nothing.
- b ~= 0.5 (more generally, < 1) is sub-linear.  Here, the GPU is filling up, with parallelism absorbing some work.
- b ~= 1 is linear.  Here the GPU is fully saturated, and runtime grows proportionally with N.
- b > 1 is super-linear (worse than linear growth).  Here the kernel work dominates.

| Scaling: CUDA |
|----------------|
| <img width="691" height="381" alt="Screenshot 2026-05-14 at 9 53 35 AM" src="https://github.com/user-attachments/assets/be385135-c75d-4903-b09d-1e094d2acb8f" /> |

My conclusions are as follows:
- N=5000 is within the sub-linear zone.  The GPU is filling up, but parallelism partially absorbs extra work.
- N=8192 appears to be the boundary between sub-linear and super-linear.
- The GPU I ran on was an L4 whose L2 cache has 48MB.  At the largest tested N=131k, the bodies take up a total of 11MB.  For the math, Config.h defines each body as 10 double arrays (8 bytes each) and 1 int array (4 bytes), meaning 84 bytes per body.  84 bytes * 131,072 bodies ~= 11 MB.  So this fits comfortably within cache, and memory pressure isn't the likely culprit.  The most likely culprit is the O(N^2) kernel work fom compute_acceleration_kernel overwhelming the parallelism.  From nvidia-smi commands, I discovered I have 7,424 CUDA cores at the time of the run.  With N=131,072, the GPU launches 131,072 threads across 7,424 cores.  Each thread loops over all other bodies, meaning ~17 billion pairwise force calculations per step, serialized inside each thread's inner loop. The GPU can parallelize across N outer threads, but cannot parallelize the inner loops (as an example, for (int j = 0; j < N)). As N grows, this inner loop gets longer.

# Key Profiling Results
I performed VTune on the serial and 16-threaded OpenMP versions to see which function(s) take up the most time, and how OpenMP helps.  For brevity, the OpenMP table results is shortened to focus on the top-3 hotspots.

## Serial
| Function                                  | CPU Time | Effective Time | Spin Time | Overhead Time | Full Function Signature |
|-------------------------------------------|----------|----------------|-----------|---------------|--------------------------|
| compute_acceleration_serial               | 2.073s   | 2.073s         | 0s        | 0s            | compute_acceleration_serial(std::vector<Body>&) |
| Vec3::operator*                           | 1.469s   | 1.469s         | 0s        | 0s            | Vec3::operator*(double) const |
| Vec3::operator+                           | 0.918s   | 0.918s         | 0s        | 0s            | Vec3::operator+(const Vec3&) const |
| check_collisions_serial                   | 0.770s   | 0.770s         | 0s        | 0s            | check_collisions_serial(std::vector<Body>&) |
| Vec3::norm                                | 0.730s   | 0.730s         | 0s        | 0s            | Vec3::norm() const |
| Vec3::operator-                           | 0.560s   | 0.560s         | 0s        | 0s            | Vec3::operator-(const Vec3&) const |
| Vec3::operator-                           | 0.280s   | 0.280s         | 0s        | 0s            | Vec3::operator-(const Vec3&) const |
| std::vector::size                         | 0.230s   | 0.230s         | 0s        | 0s            | std::vector<Body>::size() const |
| Vec3::norm                                | 0.230s   | 0.230s         | 0s        | 0s            | Vec3::norm() const |
| std::vector::operator[]                   | 0.160s   | 0.160s         | 0s        | 0s            | std::vector<Body>::operator[](size_t) |
| std::ostream::operator<< (double)         | 0.130s   | 0.130s         | 0s        | 0s            | std::ostream::operator<<(double) |
| std::ostream::operator<< (double)         | 0.080s   | 0.080s         | 0s        | 0s            | std::ostream::operator<<(double) |
| std::ostream::operator<< (double)         | 0.050s   | 0.050s         | 0s        | 0s            | std::ostream::operator<<(double) |
| std::ostream::operator<< (double)         | 0.050s   | 0.050s         | 0s        | 0s            | std::ostream::operator<<(double) |
| std::ostream::operator<< (unsigned long)  | 0.020s   | 0.020s         | 0s        | 0s            | std::ostream::operator<<(unsigned long) |
| std::operator<< (char*)                   | 0.010s   | 0.010s         | 0s        | 0s            | std::operator<<(std::ostream&, const char*) |
| std::ostream::operator<< (double)         | 0.010s   | 0.010s         | 0s        | 0s            | std::ostream::operator<<(double) |             

## Openmp
| Function                               | CPU Time | Effective Time | Spin Time | Overhead Time | Full Function Signature |
|----------------------------------------|----------|----------------|-----------|---------------|--------------------------|
| compute_acceleration_openmp._omp_fn.0  | 8.894s   | 8.894s         | 0s        | 0s            | compute_acceleration_openmp._omp_fn.0 |
| gomp_simple_barrier_wait               | 3.208s   | 0.060s         | 3.148s    | 0s            | gomp_simple_barrier_wait |
| check_collisions_openmp                | 0.909s   | 0.909s         | 0s        | 0s            | check_collisions_openmp(std::vector<Body, std::allocator<Body>>&) |
| ...                          | ...  | ...       | ...      | ...        | ... |

My conclusions are as follows:
- It is evident that `compute_acceleration` takes up the most time and therefore is the main bottleneck among the three main functions I am trying to improve.
- The tables show OpenMP helps reduce `compute_acceleration` time.  Notably, it takes 8.894s summed over all 16 threads, or 8.894/16 = 0.555875s for 1 thread.  Note that this calculation assumes an equal load balance, which is not true in reality (nevertheless, this simplified estimate of 0.555875s for 1 thread indicates a general reduction from the serial's 2.703s CPU time).  I know there is unequal load balance because gomp_simple_barrier_wait is nontrivial (3.208s).  This is likely happening because the work per body is uneven, like one body having closer nearby neighbors.  VTune even highlights this in its conclusion, quoting: "Imbalance or Serial Spinning: 3.158s... This can be caused by a load imbalance...".  For a future OpenMP version, I suggest dynamic scheduling in the integrate function's pragma directives!
- It may be initially alarming that `check_collisions_openmp` shows up only on the 16-threaded OpenMP hotspot.  When translated back to time for 1 thread, however, we get 0.909/16 = 0.0568125s per thread (with the caveat, similar to `compute_acceleration`, that this is a simplified estimate), which is smaller than any of the values in the serial hotspot table.  This therefore is not an immediate concern.

Taken together, these trends confirm that OpenMP parallelization is able to target the dominant bottleneck, with the primary remaining inefficiency being load balancing.

# Conclusions
In conclusion, this project highlights the strengths and imitations of parallelization strategies for an N-body orbital simulation.  From the serial implementation, the dominant cost comes from the compute_acceleration function, as confirmed by VTune.  This is expected: every body interacts with every other body, leading to quadratic scaling that quickly dominates execution time as N grows.  Of all the approaches explored, CUDA delivered the fastest time.  For the CPU, Kokkos OpenMP delivered the fastest initial time, but was later matched by OpenMP with more threads due to the amount of deep-copying data movement in Kokkos' implementation.  OpenMP provided reliable speedups with relatively low implementation complexity, but diminished in returns beyond 8-16 threads.  MPI saw additional communication overhead that reduced performance relative to OpenMP.  Overall, the different strategies show the importance of algorithmic setup, fraction of parallel/serialization, and data structures on resulting performance.
