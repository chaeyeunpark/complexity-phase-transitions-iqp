#include <Kokkos_Core.hpp>

#include <ctime>
#include <iostream>
#include <random>
#include <cmath>
#include <mpi.h>

#include "npy.hpp"
#include "json.hpp"

template <typename T>
using HostView = Kokkos::View<T, Kokkos::LayoutLeft, Kokkos::HostSpace>;
template <typename T>
using DeviceView = Kokkos::View<T, Kokkos::LayoutLeft>;

double squared_norm(Kokkos::complex<double> val) {
	return val.real()*val.real() + val.imag()*val.imag();
}

double compute_probability(const size_t num_qubits, const HostView<double**>& J, const HostView<double*>& theta) {
	const size_t dim = 1UL << num_qubits;

	DeviceView<double**> J_d("J_d", num_qubits, num_qubits);
	Kokkos::deep_copy(J_d, J);

	DeviceView<double*> theta_d("theta_d", num_qubits);
	Kokkos::deep_copy(theta_d, theta);

	Kokkos::complex<double> result = 0.0;
	Kokkos::parallel_reduce(Kokkos::RangePolicy<size_t>(0UL, dim), KOKKOS_LAMBDA(const size_t idx, Kokkos::complex<double>& sum) {
		double e = 0;
		for(size_t i = 0; i < num_qubits - 1; i++) {
			for(size_t j = i+1; j < num_qubits; j++) {
				int zi = 1-2*int((idx >> i) & 1L);
				int zj = 1-2*int((idx >> j) & 1L);

				e += J_d(i, j) * zi * zj;
			}
		}

		for(size_t i = 0; i < num_qubits; i++) {
			int zi = 1-2*int((idx >> i) & 1L);
			e += theta_d(i) * zi;
		}

		sum += Kokkos::complex(Kokkos::cos(e), -Kokkos::sin(e));
	}, result);

	// std::cout << result << "\t" << squared_norm(result) << "\t" << double(dim) << std::endl;

	return squared_norm(result) / std::pow(double(dim),2);
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	Kokkos::initialize(argc, argv);

	{
		int mpi_rank, mpi_size;
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

		if(argc != 3) {
			printf("Usage: %s [N] [alpha]\n", argv[0]);
			return 1;
		}

		const size_t N = std::stol(argv[1]);
		const double alpha = std::stod(argv[2]);

		const size_t total_iter = [N]{
			if (N <= 28) {
				return 1024*64*8;
			} else {
				return 1024*64*2;
			}
		}();

		const double q = alpha*std::log(double(N))/double(N);

		if (mpi_rank == 0) {
			nlohmann::json params_in = {
				{"N", N},
				{"alpha", alpha},
				{"q", q}
			};
			std::ofstream fout("params_in.json");

			fout << params_in.dump();
		}

		MPI_Barrier(MPI_COMM_WORLD);

		std::vector<double> probs;
		probs.reserve(total_iter);

		std::random_device rd;
		std::mt19937_64 re{rd() + mpi_rank};
		std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
		std::uniform_real_distribution<double> param_dist(0.0, 2.0*M_PI);

		for(size_t iter = mpi_rank; iter < total_iter; iter += mpi_size) {
			printf("Processing iter=%lu in mpi_rank=%d\n", iter, mpi_rank);
			HostView<double**> J("J_h", N, N);
			HostView<double*> theta("theta_h", N);

			for(size_t i = 0; i < N; i++) {
				for(size_t j = 0; j < N; j++) {
					J(i, j) = 0.0;
				}
			}

			for(size_t i = 0; i < N - 1; i++) {
				for(size_t j = i+1; j < N; j++) {
					if(prob_dist(re) < q) {
						J(i, j) = param_dist(re);
					} 
				}
			}

			for(size_t i = 0; i < N; i++) {
				theta(i) = param_dist(re);
			}

			const double prob = compute_probability(N, J, theta);
			probs.push_back(prob);
		}

		npy::npy_data_ptr<double> to_save;
		to_save.data_ptr = probs.data();
		to_save.shape = {probs.size()};
		
		char filename[255];
		sprintf(filename, "probs_%03d.npy", mpi_rank);
		const std::string path(filename);
		write_npy(path, to_save);
	} // scope guard
	Kokkos::finalize();
	MPI_Finalize();
	return 0;
}
