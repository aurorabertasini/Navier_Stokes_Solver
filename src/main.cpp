#include "../include/Stokes.hpp"
#include "../include/Incremental_stokes.hpp"
#include "../include/MonolithicNavierStokes.hpp"
#include "../include/ChorinTemam.hpp"
#include "../include/IncrementalChorinTemam.hpp"
#include "../include/ConfigReader.hpp"

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int mpi_rank =
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
        std::cout << "Welcome to the Navier-Stokes solver" << std::endl;
    }

    double cylinder_radius = 0.1;

    double uM = 1.5;

    ConfigReader configReader;

    std::filesystem::path mesh2DPath = configReader.getMesh2DPath();
    std::filesystem::path mesh3DPath = configReader.getMesh3DPath();
    int degreeVelocity = configReader.getDegreeVelocity();
    int degreePressure = configReader.getDegreePressure();
    double simulationPeriod = configReader.getSimulationPeriod();
    double timeStep = configReader.getTimeStep();
    double Re = configReader.getRe();

    double nu = (uM * cylinder_radius) / Re;

    int choice;

    if (mpi_rank == 0)
    {
        std::cout << "Please choose the problem to solve:" << std::endl;
        std::cout << "(1) Steady Navier-Stokesm Problem 2D" << std::endl;
        std::cout << "(2) Steady Navier-Stokesm Problem 3D" << std::endl;
        std::cout << "(3) Monolithic Time Dependent Navier-Stokesm Problem 2D" << std::endl;
        std::cout << "(4) Monolithic Time Dependent Navier-Stokesm Problem 3D" << std::endl;
        std::cout << "(5) Chorin-Temam Time Dependent Navier-Stokesm Problem 2D" << std::endl;
        std::cout << "(6) Chorin-Temam Time Dependent Navier-Stokesm Problem 3D" << std::endl;
        std::cout << "(7) Incremental Chorin-Temam Time Dependent Navier-Stokesm Problem 2D" << std::endl;
        std::cout << "(8) Incremental Chorin-Temam Time Dependent Navier-Stokesm Problem 3D" << std::endl;
        std::cout << std::endl;
        std::cout << "Enter your choice: ";

        while (choice < 1 || choice > 8)
        {
            std::cin >> choice;
            if (choice < 1 || choice > 8)
            {
                std::cout << "Invalid choice. Please enter a valid choice: ";
            }
        }
    }

    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    switch (choice)
    {
    case 1:
    {
        std::cout << "Solving the Steady Navier-Stokesm Problem 2D" << std::endl;
        Stokes stokes(mesh2DPath, degreeVelocity, degreePressure, Re);

        stokes.setup();
        stokes.assemble();
        stokes.solve();
        stokes.output();

        IncrementalStokes incremental(mesh2DPath, degreeVelocity, degreePressure, Re);
        incremental.set_initial_conditions(stokes.get_solution());
        incremental.setup();
        incremental.solve();
        incremental.output();
        incremental.compute_lift_drag();
        break;
    }
    case 2:
    {
        std::cout << "Not Available :(" << std::endl;
        exit(0);
    }
    case 3:
    {
        MonolithicNavierStokes monolithicNavierStokes(mesh2DPath, degreeVelocity, degreePressure, simulationPeriod, timeStep, 1, Re);
        monolithicNavierStokes.setup();
        monolithicNavierStokes.solve();
        break;
    }
    case 4:
    {
        std::cout << "Not Available :(" << std::endl;
        exit(0);
    }
    case 5:
    {
        ChorinTemam chorinTemam(mesh2DPath, degreeVelocity, degreePressure, simulationPeriod, timeStep, Re);
        chorinTemam.run();
        break;
    }
    case 6:
    {
        std::cout << "Not Available :(" << std::endl;
        exit(0);
    }
    case 7:
    {
        std::cout << "Not Available :(" << std::endl;
        exit(0);
    }
    case 8:
    {
        std::cout << "Not Available :(" << std::endl;
        exit(0);
    }

        return 0;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (mpi_rank == 0)
    {
        std::cout << "Elapsed time: " << elapsed.count() << " s\n";
        std::cout << std::endl
                  << "THE END" << std::endl;
    }
}
