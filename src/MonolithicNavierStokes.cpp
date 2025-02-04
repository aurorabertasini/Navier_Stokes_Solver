#include "../include/MonolithicNavierStokes.hpp"

template <unsigned int dim>
void MonolithicNavierStokes<dim>::setup()
{
    // Create the mesh.
    {
        pcout << "Initializing the mesh" << std::endl;

        Triangulation<dim> mesh_serial;

        GridIn<dim> grid_in;
        grid_in.attach_triangulation(mesh_serial);

        std::ifstream grid_in_file(mesh_file_name);
        grid_in.read_msh(grid_in_file);

        GridTools::partition_triangulation(mpi_size, mesh_serial);
        const auto construction_data = TriangulationDescription::Utilities::
            create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
        mesh.create_triangulation(construction_data);

        pcout << "  Number of elements = " << mesh.n_global_active_cells()
              << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the finite element space.
    {
        pcout << "Initializing the finite element space" << std::endl;

        const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
        const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);

        fe = std::make_unique<FESystem<dim>>(
            fe_scalar_velocity,
            dim,
            fe_scalar_pressure,
            1);

        pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree
              << std::endl;
        pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree
              << std::endl;
        pcout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

        quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);

        pcout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;

        quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

        pcout << "  Quadrature points per face = " << quadrature_face->size()
              << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the DoF handler.
    {
        pcout << "Initializing the DoF handler" << std::endl;

        dof_handler.reinit(mesh);
        dof_handler.distribute_dofs(*fe);

        // We want to reorder DoFs so that all velocity DoFs come first, and then
        // all pressure DoFs.
        std::vector<unsigned int> block_component(dim + 1, 0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise(dof_handler, block_component);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        std::vector<types::global_dof_index> dofs_per_block =
            DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
        const unsigned int n_u = dofs_per_block[0];
        const unsigned int n_p = dofs_per_block[1];

        block_owned_dofs.resize(2);
        block_relevant_dofs.resize(2);
        block_owned_dofs[0] = locally_owned_dofs.get_view(0, n_u);
        block_owned_dofs[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);
        block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
        block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

        pcout << "  Number of DoFs: " << std::endl;
        pcout << "    velocity = " << n_u << std::endl;
        pcout << "    pressure = " << n_p << std::endl;
        pcout << "    total    = " << n_u + n_p << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the linear system.
    {
        pcout << "Initializing the linear system" << std::endl;
        pcout << "  Initializing the sparsity pattern" << std::endl;

        Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
        for (unsigned int c = 0; c < dim + 1; ++c)
        {
            for (unsigned int d = 0; d < dim + 1; ++d)
            {
                if (c == dim && d == dim) // pressure-pressure term
                    coupling[c][d] = DoFTools::none;
                else // other combinations
                    coupling[c][d] = DoFTools::always;
            }
        }

        TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs,
                                                        MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity);
        sparsity.compress();

        // Do the same for the velocity mass term.
        for (unsigned int c = 0; c < dim + 1; ++c)
        {
            for (unsigned int d = 0; d < dim + 1; ++d)
            {
                if (c == dim || d == dim) // terms with pressure
                    coupling[c][d] = DoFTools::none;
                else // terms with no pressure
                    coupling[c][d] = DoFTools::always;
            }
        }

        TrilinosWrappers::BlockSparsityPattern velocity_mass_sparsity(
            block_owned_dofs, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, coupling,
                                        velocity_mass_sparsity);
        velocity_mass_sparsity.compress();

        // Do the same for the pressure mass term.
        TrilinosWrappers::BlockSparsityPattern pressure_mass_sparsity(
            block_owned_dofs, MPI_COMM_WORLD);
        if (true)
        {
            for (unsigned int c = 0; c < dim + 1; ++c)
            {
                for (unsigned int d = 0; d < dim + 1; ++d)
                {
                    if (c == dim && d == dim) // terms with only pressure
                        coupling[c][d] = DoFTools::always;
                    else // terms with velocity
                        coupling[c][d] = DoFTools::none;
                }
            }

            DoFTools::make_sparsity_pattern(dof_handler, coupling,
                                            pressure_mass_sparsity);
            pressure_mass_sparsity.compress();
        }

        pcout << "  Initializing the matrices" << std::endl;

        system_matrix.reinit(sparsity);
        velocity_mass.reinit(velocity_mass_sparsity);
        pressure_mass.reinit(pressure_mass_sparsity);

        pcout << "  Initializing the system right-hand side" << std::endl;
        system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);

        pcout << "  Initializing the solution vector" << std::endl;
        solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
        solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
        // solution.update_ghost_values();
    }
}

template <unsigned int dim>
void MonolithicNavierStokes<dim>::assemble_base_matrix()
{
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe,
                            *quadrature,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_system_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> previous_velocity_values(n_q);
    std::vector<double> previous_velocity_divergence(n_q);
    // std::cout << "system_matrix initialized to 0.0" << std::endl;
    FullMatrix<double> velocity_mass_cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> pressure_mass_cell_matrix(dofs_per_cell, dofs_per_cell);
    // std::cout << "pressure_mass_cell_matrix initialized to 0.0" << std::endl;
    velocity_mass = 0.0;
    pressure_mass = 0.0;
    system_matrix = 0.0;
    // std::cout << "pressure_mass initialized to 0.0" << std::endl;

    // std::cout << "Starting assembly of base matrix" << std::endl;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        // std::cout << "fe_values reinitialized for cell" << std::endl;

        std::vector<double> div_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
        std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
        std::vector<double> phi_p(dofs_per_cell);

        pressure_mass_cell_matrix = 0.0;
        velocity_mass_cell_matrix = 0.0;
        cell_system_matrix = 0.0;

        fe_values[velocity].get_function_values(solution, previous_velocity_values);
        fe_values[velocity].get_function_divergences(solution, previous_velocity_divergence);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            for (unsigned int k = 0; k < dofs_per_cell; k++)
            {
                div_phi_u[k] = fe_values[velocity].divergence(k, q);
                grad_phi_u[k] = fe_values[velocity].gradient(k, q);
                phi_u[k] = fe_values[velocity].value(k, q);
                phi_p[k] = fe_values[pressure].value(k, q);
            }
            // std::cout << "Values at quadrature point " << q << " obtained" << std::endl;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // mass component
                    cell_system_matrix(i, j) += scalar_product(phi_u[i], phi_u[j]) / deltat * fe_values.JxW(q);
                    velocity_mass_cell_matrix(i, j) += scalar_product(phi_u[i], phi_u[j]) / deltat * fe_values.JxW(q);
                    // stiffness component
                    cell_system_matrix(i, j) += nu * scalar_product(grad_phi_u[i], grad_phi_u[j]) * fe_values.JxW(q);
                    // Pressure term in the momentum equation
                    cell_system_matrix(i, j) -= phi_p[i] * div_phi_u[j] * fe_values.JxW(q);
                    // Pressure term in the continuity equation
                    cell_system_matrix(i, j) -= phi_p[j] * div_phi_u[i] * fe_values.JxW(q);
                    // Non-linear term
                    cell_system_matrix(i, j) += scalar_product(grad_phi_u[j] * previous_velocity_values[q], phi_u[i]) * fe_values.JxW(q);
                    // skew-symmetric term
                    cell_system_matrix(i, j) += 0.5 * previous_velocity_divergence[q] * scalar_product(phi_u[i], phi_u[j]) * fe_values.JxW(q);

                    pressure_mass_cell_matrix(i, j) += fe_values[pressure].value(j, q) *
                                                       fe_values[pressure].value(i, q) /
                                                       nu * fe_values.JxW(q);
                }
            }
            // std::cout << "cell_system_matrix and pressure_mass_cell_matrix updated for quadrature point " << q << std::endl;
        }
        cell->get_dof_indices(dof_indices);
        // std::cout << "dof_indices obtained for cell" << std::endl;

        system_matrix.add(dof_indices, cell_system_matrix);
        velocity_mass.add(dof_indices, velocity_mass_cell_matrix);
        pressure_mass.add(dof_indices, pressure_mass_cell_matrix);
        // std::cout << "system_matrix and pressure_mass updated for cell" << std::endl;
    }
    system_matrix.compress(VectorOperation::add);
    pressure_mass.compress(VectorOperation::add);
    velocity_mass.compress(VectorOperation::add);
    // std::cout << "system_matrix and pressure_mass compressed" << std::endl;

    // std::cout << "Finished assembly of base matrix" << std::endl;
}

template <unsigned int dim>
void MonolithicNavierStokes<dim>::assemble_rhs(const double &time)
{
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();
    const unsigned int n_q_face = quadrature_face->size();

    FEValues<dim> fe_values(*fe,
                            *quadrature,
                            update_values | update_gradients | update_quadrature_points |
                                update_JxW_values);

    FEFaceValues<dim> fe_face_values(*fe,
                                     *quadrature_face,
                                     update_values |
                                         update_normal_vectors |
                                         update_JxW_values);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<Tensor<1, dim>> previous_velocity_values(n_q);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_rhs = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        fe_values[velocity].get_function_values(solution, previous_velocity_values);

        cell_rhs = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            // Compute f(tn+1)
            Vector<double> f_new_loc(dim);
            forcing_term.set_time(time);
            forcing_term.vector_value(fe_values.quadrature_point(q),
                                      f_new_loc);
            Tensor<1, dim> forcing_term_new_tensor;
            for (unsigned int d = 0; d < dim; ++d)
                forcing_term_new_tensor[d] = f_new_loc[d];

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                cell_rhs(i) += scalar_product(previous_velocity_values[q],
                                              fe_values[velocity].value(i, q)) *
                               fe_values.JxW(q) / deltat;
            }
        }

        if (cell->at_boundary())
        {
            for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
                if (cell->face(f)->at_boundary() &&
                    cell->face(f)->boundary_id() == 1)
                {

                    fe_face_values.reinit(cell, f);

                    for (unsigned int q = 0; q < n_q_face; ++q)
                    {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                            cell_rhs(i) +=
                                -p_out *
                                scalar_product(fe_face_values.normal_vector(q),
                                               fe_face_values[velocity].value(i,
                                                                              q)) *
                                fe_face_values.JxW(q);
                        }
                    }
                }
            }
        }
        cell->get_dof_indices(dof_indices);
        system_rhs.add(dof_indices, cell_rhs);
    }

    system_rhs.compress(VectorOperation::add);

    // We apply boundary conditions to the algebraic system.
    {
        std::map<types::global_dof_index, double> boundary_values;
        std::map<types::boundary_id, const Function<dim> *> boundary_functions;

        ComponentMask mask;

        if constexpr (dim == 2)
            mask = ComponentMask({true, true, false});
        else if constexpr (dim == 3)
            mask = ComponentMask({true, true, true, false});

        boundary_functions[0] = &inlet_velocity;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_functions,
                                                 boundary_values,
                                                 mask);

        boundary_functions.clear();
        Functions::ZeroFunction<dim> zero_function(dim + 1);
        boundary_functions[2] = &zero_function;
        boundary_functions[3] = &zero_function;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_functions,
                                                 boundary_values,
                                                 mask);

        MatrixTools::apply_boundary_values(
            boundary_values, system_matrix, solution_owned, system_rhs, false);
    }
}

template <unsigned int dim>
void MonolithicNavierStokes<dim>::solve_time_step(int precond_type, bool use_ilu,
                                                  double &elapsed_time, int &gmres_iters,
                                                  double &precond_construct_time)
{
    double alpha = 1;
    unsigned int maxiter_inner = 10000;
    double tol_inner = 1e-5;

    SolverControl solver_control(1000000, 1e-7);
    SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

    std::shared_ptr<BlockPrecondition> block_precondition;

    auto precond_start = std::chrono::high_resolution_clock::now();

    switch (precond_type)
    {
    case 1:
    {
        auto block_diag_precondition = std::make_shared<PreconditionBlockDiagonal>();
        block_diag_precondition->initialize(
            system_matrix.block(0, 0), pressure_mass.block(1, 1),
            maxiter_inner, tol_inner, use_ilu);
        block_precondition = block_diag_precondition;
        break;
    }
    case 2:
    {
        auto simple_precondition = std::make_shared<PreconditionSIMPLE>();
        simple_precondition->initialize(
            system_matrix.block(0, 0), system_matrix.block(1, 0),
            system_matrix.block(0, 1), solution_owned, alpha,
            maxiter_inner, tol_inner, use_ilu);
        block_precondition = simple_precondition;
        break;
    }
    case 3:
    {

        auto asimple_precondition = std::make_shared<PreconditionaSIMPLE>();
        asimple_precondition->initialize(
            system_matrix.block(0, 0), system_matrix.block(1, 0),
            system_matrix.block(0, 1), solution_owned, alpha,
            use_inner_solver, maxiter_inner, tol_inner, use_ilu);
        block_precondition = asimple_precondition;

        break;
    }
    case 4:
    {
        auto yosida_precondition = std::make_shared<PreconditionYosida>();
        yosida_precondition->initialize(
            system_matrix.block(0, 0), system_matrix.block(1, 0),
            system_matrix.block(0, 1), velocity_mass.block(0, 0),
            solution_owned, maxiter_inner, tol_inner, use_ilu);
        block_precondition = yosida_precondition;
        break;
    }
    case 0:
    {
        block_precondition = std::make_shared<PreconditionIdentity>();
        break;
    }
    default:
        Assert(false, ExcNotImplemented());
    }

    auto precond_end = std::chrono::high_resolution_clock::now();
    precond_construct_time = std::chrono::duration<double>(precond_end - precond_start).count();

    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(system_matrix, solution_owned, system_rhs, *block_precondition);
    auto end = std::chrono::high_resolution_clock::now();
    pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;

    elapsed_time = std::chrono::duration<double>(end - start).count();
    gmres_iters = solver_control.last_step();

    solution = solution_owned;
}

template <unsigned int dim>
void MonolithicNavierStokes<dim>::run_with_preconditioners()
{
    pcout << "===============================================" << std::endl;
    pcout << "Running Navier-Stokes solver with multiple preconditioners" << std::endl;
    pcout << "===============================================" << std::endl;

    struct SolveResult
    {
        std::string precond_type;
        int gmres_first_dt;
        double time_first_dt;
        double total_solve_time;
        double precond_construct_time;
        bool use_ilu;
    };

    std::vector<SolveResult> results;

    for (int precond_type = 2; precond_type < 5; ++precond_type)
    {
        for (bool use_ilu : {true, false})
        {
            if (precond_type == 3)
                for (bool use_inner_solver_ : {true, false})
                {
                    use_inner_solver = use_inner_solver_;
                    setup(); // Reset everything before each solve

                    double first_dt_solve_time = 0.0;
                    int first_dt_gmres_iters = 0;
                    double total_solve_time = 0.0;
                    double precond_construct_time = 0.0;

                    auto start_total = std::chrono::high_resolution_clock::now();

                    solve(precond_type, use_ilu, first_dt_solve_time, first_dt_gmres_iters, total_solve_time, precond_construct_time);

                    auto end_total = std::chrono::high_resolution_clock::now();
                    total_solve_time = std::chrono::duration<double>(end_total - start_total).count();

                    std::string name = std::to_string(precond_type) + " " + std::to_string(use_inner_solver);

                    results.push_back({name, first_dt_gmres_iters, first_dt_solve_time,
                                       total_solve_time, precond_construct_time, use_ilu});
                }else
                { 
                    setup(); // Reset everything before each solve

                    double first_dt_solve_time = 0.0;
                    int first_dt_gmres_iters = 0;
                    double total_solve_time = 0.0;
                    double precond_construct_time = 0.0;

                    auto start_total = std::chrono::high_resolution_clock::now();

                    solve(precond_type, use_ilu, first_dt_solve_time, first_dt_gmres_iters, total_solve_time, precond_construct_time);

                    auto end_total = std::chrono::high_resolution_clock::now();
                    total_solve_time = std::chrono::duration<double>(end_total - start_total).count();

                    std::string name = std::to_string(precond_type) + " " + std::to_string(use_inner_solver);

                    results.push_back({name, first_dt_gmres_iters, first_dt_solve_time,
                                       total_solve_time, precond_construct_time, use_ilu});}
        }
    }

    // Print results table
    pcout << "============================================================================================" << std::endl;
    pcout << "Preconditioner Performance Summary" << std::endl;
    pcout << "============================================================================================" << std::endl;
    pcout << std::setw(15) << "Precond Type"
          << std::setw(20) << "GMRES 1st DT"
          << std::setw(20) << "Time 1st DT (s)"
          << std::setw(25) << "Total Solve Time (s)"
          << std::setw(25) << "Precond Construct (s)"
          << std::setw(15) << "Use ILU" << std::endl;
    pcout << "--------------------------------------------------------------------------------------------" << std::endl;

    for (const auto &result : results)
    {
        pcout << std::setw(15) << result.precond_type
              << std::setw(20) << result.gmres_first_dt
              << std::setw(20) << result.time_first_dt
              << std::setw(25) << result.total_solve_time
              << std::setw(25) << result.precond_construct_time
              << std::setw(15) << (result.use_ilu ? "True" : "False") << std::endl;
    }

    pcout << "============================================================================================" << std::endl;
}

// Modify solve to accept preconditioner parameters

template <unsigned int dim>
void MonolithicNavierStokes<dim>::solve(int precond_type, bool use_ilu, double &first_dt_solve_time,
                                        int &first_dt_gmres_iters, double &total_solve_time, double &precond_construct_time)
{
    pcout << "Solving for preconditioner type " << precond_type << " with ILU = " << (use_ilu ? "True" : "False") << std::endl;

    time = 0.0;

    // Apply the initial condition.
    {
        VectorTools::interpolate(dof_handler, initial_condition, solution_owned);
        solution = solution_owned;
    }

    unsigned int time_step = 0;
    auto start_total = std::chrono::high_resolution_clock::now();

    while (time < deltat)
    {
        time += deltat;
        ++time_step;

        pcout << "n = " << std::setw(3) << time_step << std::endl
              << std::flush;

        assemble_base_matrix();
        assemble_rhs(time);

        double elapsed_time;
        int gmres_iters;
        double construct_time;

        solve_time_step(precond_type, use_ilu, elapsed_time, gmres_iters, construct_time);

        if (time_step == 1)
        {
            first_dt_solve_time = elapsed_time;
            first_dt_gmres_iters = gmres_iters;
            precond_construct_time = construct_time;
        }

        compute_lift_drag();
        // output(time_step);
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    total_solve_time = std::chrono::duration<double>(end_total - start_total).count();
}

template <unsigned int dim>
void MonolithicNavierStokes<dim>::output(const unsigned int &time_step)
{
    DataOut<dim> data_out;

    // Define correct interpretation for velocity and pressure
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

    // Make sure names match interpretation (e.g., 3D: velocity_x, velocity_y, velocity_z)
    std::vector<std::string> names;
    for (unsigned int i = 0; i < dim; ++i)
        names.push_back("velocity");
    names.push_back("pressure");

    // Add data vector
    data_out.add_data_vector(dof_handler, solution, names, data_component_interpretation);

    // Add partitioning information
    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<float> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    // Generate patches
    // solution.update_ghost_values();

    data_out.build_patches();

    // std::string numProcessors = std::to_string(mpi_size);
    // numProcessors += (mpi_size == 1) ? "_processor" : "_processors";

    std::string output_dir = get_output_directory();

    // Set output file name

    // Write output to VTU/PVTU
    // data_out.write_vtu_with_pvtu_record("./output-IncrementalStokes",
    //                                     time_step,
    //                                     MPI_COMM_WORLD,3);

    data_out.write_vtu_with_pvtu_record(
        output_dir, "output_", time_step, MPI_COMM_WORLD, 3);
}

template <unsigned int dim>
void MonolithicNavierStokes<dim>::compute_lift_drag()
{

    QGauss<dim - 1> face_quadrature_formula(3);
    const int n_q_points = face_quadrature_formula.size();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    std::vector<double> pressure_values(n_q_points);
    std::vector<Tensor<2, dim>> velocity_gradients(n_q_points);

    Tensor<1, dim> grad_u_tau;
    Tensor<1, dim> normal_vector;
    Tensor<1, dim> tangent_vector;

    Tensor<2, dim> fluid_stress;
    Tensor<2, dim> fluid_pressure;
    Tensor<1, dim> forces;

    FEFaceValues<dim> fe_face_values(*fe, face_quadrature_formula, update_values | update_quadrature_points | update_gradients | update_JxW_values | update_normal_vectors);

    double drag = 0;
    double lift = 0;

    double u_max = inlet_velocity.get_u_max();

    double u_mean = 2 * u_max / 3;

    double D = 0.1;

    double coefficient = 2 / (rho * u_mean * u_mean * D);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (cell->at_boundary())
        {

            for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
                fe_face_values.reinit(cell, f);
                std::vector<Point<dim>> q_points = fe_face_values.get_quadrature_points();

                if (cell->face(f)->boundary_id() == 4)
                {
                    fe_face_values[velocities].get_function_gradients(solution, velocity_gradients);
                    fe_face_values[pressure].get_function_values(solution, pressure_values);

                    for (int q = 0; q < n_q_points; q++)
                    {
                        normal_vector = -fe_face_values.normal_vector(q);

                        fluid_pressure.clear(); // Initialize to zero
                        fluid_pressure[0][0] = pressure_values[q];
                        fluid_pressure[1][1] = pressure_values[q];

                        fluid_stress = nu * velocity_gradients[q] - fluid_pressure;

                        forces = fluid_stress * normal_vector * fe_face_values.JxW(q);

                        drag += coefficient * forces[0];
                        lift += coefficient * forces[1];
                    }
                }
            }
        }
    }

    // std::cout << "Drag = " << drag << std::endl;
    // std::cout << "Lift = " << lift << std::endl;

    Point<dim> p1, p2;
    p1[0] = 0.15;
    p1[1] = 0.2;
    p2[0] = 0.25;
    p2[1] = 0.2;
    Vector<double> solution_values1(dim + 1);
    Vector<double> solution_values2(dim + 1);

    bool p1_available = true;
    bool p2_available = true;

    try
    {
        VectorTools::point_value(dof_handler, solution, p1, solution_values1);
    }
    catch (const dealii::VectorTools::ExcPointNotAvailableHere &)
    {
        p1_available = false;
    }

    try
    {
        VectorTools::point_value(dof_handler, solution, p2, solution_values2);
    }
    catch (const dealii::VectorTools::ExcPointNotAvailableHere &)
    {
        p2_available = false;
    }

    if (p1_available)
    {
        MPI_Send(&solution_values1[dim], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if (p2_available)
    {
        MPI_Send(&solution_values2[dim], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Send(&lift, 1, MPI_DOUBLE, 0, 2 * mpi_rank, MPI_COMM_WORLD);
    MPI_Send(&drag, 1, MPI_DOUBLE, 0, 2 * mpi_rank + 1, MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
        MPI_Status status;
        // Ricevi i dati da chiunque abbia inviato per p1
        MPI_Recv(&solution_values1[dim], 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        int source_p1 = status.MPI_SOURCE;

        // Ricevi i dati da chiunque abbia inviato per p2
        MPI_Recv(&solution_values2[dim], 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        int source_p2 = status.MPI_SOURCE;

        double p_diff = solution_values1(dim) - solution_values2(dim);
        // std::cout << "Pressure difference between points 1 and 2: " << p_diff << std::endl;

        for (int i = 1; i < mpi_size; i++)
        {
            double lift_i, drag_i;
            MPI_Recv(&lift_i, 1, MPI_DOUBLE, i, 2 * i, MPI_COMM_WORLD, &status);
            MPI_Recv(&drag_i, 1, MPI_DOUBLE, i, 2 * i + 1, MPI_COMM_WORLD, &status);
            lift += lift_i;
            drag += drag_i;
        }

        std::string output_dir = get_output_directory();

        std::string filename = output_dir + "lift_drag_output.csv";

        FILE *pFile = fopen(filename.c_str(), "a");
        fprintf(pFile, "%f, %f,  %f, %f\n", time, drag, lift, p_diff); // 500 = 2 / (U^2 * L)     // (L = 0.1, U = 2/3 * 0.3)
        fflush(pFile);
        fclose(pFile); // Close the file after writing
    }
}

template <unsigned int dim>
std::string MonolithicNavierStokes<dim>::get_output_directory()
{
    namespace fs = std::filesystem;

    if (!fs::exists("./outputs"))
    {
        fs::create_directory("./outputs");
    }
    if constexpr (dim == 2)
    {
        if (!fs::exists("./outputs/monolithicNavierStokes2D"))
        {
            fs::create_directory("./outputs/monolithicNavierStokes2D");
        }
    }
    else if constexpr (dim == 3)
    {
        if (!fs::exists("./outputs/monolithicNavierStokes3D"))
        {
            fs::create_directory("./outputs/monolithicNavierStokes3D");
        }
    }

    std::string sub_dir_name = "outputs_reynolds_" + std::to_string(static_cast<int>(reynolds_number));
    fs::path sub_dir_path = "";
    if constexpr (dim == 2)
        sub_dir_path = "./outputs/monolithicNavierStokes2D/" + sub_dir_name + "/";
    if constexpr (dim == 3)
        sub_dir_path = "./outputs/monolithicNavierStokes3D/" + sub_dir_name + "/";
    if (!fs::exists(sub_dir_path))
    {
        fs::create_directory(sub_dir_path);
    }

    return sub_dir_path.string();
}

template class MonolithicNavierStokes<2>;
template class MonolithicNavierStokes<3>;