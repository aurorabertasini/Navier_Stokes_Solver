#include "../../include/Stokes.hpp"

void Stokes::setup()
{
    Linardo ::setup();

    // Initialize the finite element space.
    {
        pcout << "Initializing the finite element space" << std::endl;

        const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
        const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);
        fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity,
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

        // We also build a sparsity pattern for the pressure mass matrix.
        for (unsigned int c = 0; c < dim + 1; ++c)
        {
            for (unsigned int d = 0; d < dim + 1; ++d)
            {
                if (c == dim && d == dim) // pressure-pressure term
                    coupling[c][d] = DoFTools::always;
                else // other combinations
                    coupling[c][d] = DoFTools::none;
            }
        }
        TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
            block_owned_dofs, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler,
                                        coupling,
                                        sparsity_pressure_mass);
        sparsity_pressure_mass.compress();

        pcout << "  Initializing the matrices" << std::endl;
        system_matrix.reinit(sparsity);
        pressure_mass.reinit(sparsity_pressure_mass);

        pcout << "  Initializing the system right-hand side" << std::endl;
        system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
        pcout << "  Initializing the solution vector" << std::endl;
        solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
        solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
    }
}

void Stokes::assemble()
{
    pcout << "===============================================" << std::endl;
    pcout << "Assembling the system" << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();
    const unsigned int n_q_face = quadrature_face->size();

    FEValues<dim> fe_values(*fe,
                            *quadrature,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(*fe,
                                     *quadrature_face,
                                     update_values | update_normal_vectors |
                                         update_JxW_values);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_matrix = 0.0;
    system_rhs = 0.0;
    pressure_mass = 0.0;

    FEValuesExtractors::Vector velocity(0);
    FEValuesExtractors::Scalar pressure(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        cell_matrix = 0.0;
        cell_rhs = 0.0;
        cell_pressure_mass_matrix = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            Vector<double> forcing_term_loc(dim);
            forcing_term.vector_value(fe_values.quadrature_point(q),
                                      forcing_term_loc);
            Tensor<1, dim> forcing_term_tensor;
            for (unsigned int d = 0; d < dim; ++d)
                forcing_term_tensor[d] = forcing_term_loc[d];

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // Viscosity term.
                    cell_matrix(i, j) +=
                        nu *
                        scalar_product(fe_values[velocity].gradient(i, q),
                                       fe_values[velocity].gradient(j, q)) *
                        fe_values.JxW(q);

                    // Pressure term in the momentum equation.
                    cell_matrix(i, j) -= fe_values[velocity].divergence(i, q) *
                                         fe_values[pressure].value(j, q) *
                                         fe_values.JxW(q);

                    // Pressure term in the continuity equation.
                    cell_matrix(i, j) -= fe_values[velocity].divergence(j, q) *
                                         fe_values[pressure].value(i, q) *
                                         fe_values.JxW(q);

                    // Pressure mass matrix.
                    cell_pressure_mass_matrix(i, j) +=
                        fe_values[pressure].value(i, q) *
                        fe_values[pressure].value(j, q) / nu * fe_values.JxW(q);
                }

                // Forcing term.
                cell_rhs(i) += scalar_product(forcing_term_tensor,
                                              fe_values[velocity].value(i, q)) *
                               fe_values.JxW(q);
            }
        }

        // Boundary integral for Neumann BCs.
        if (cell->at_boundary())
        {
            for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
                if (cell->face(f)->at_boundary() &&
                    cell->face(f)->boundary_id() == 2)
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

        system_matrix.add(dof_indices, cell_matrix);
        system_rhs.add(dof_indices, cell_rhs);
        pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
    }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    pressure_mass.compress(VectorOperation::add);

    // Dirichlet boundary conditions.
    {
        std::map<types::global_dof_index, double> boundary_values;
        std::map<types::boundary_id, const Function<dim> *> boundary_functions;

        boundary_functions[1] = &inlet_velocity;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_functions,
                                                 boundary_values,
                                                 ComponentMask(
                                                     {true, true, false, false}));

        boundary_functions.clear();
        Functions::ZeroFunction<dim> zero_function(dim + 1);
        boundary_functions[3] = &zero_function;
        boundary_functions[4] = &zero_function;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_functions,
                                                 boundary_values,
                                                 ComponentMask(
                                                     {true, true, false, false}));

        MatrixTools::apply_boundary_values(
            boundary_values, system_matrix, solution_owned, system_rhs, false);
    }
}

void Stokes::solve()
{
    pcout << "===============================================" << std::endl;

    SolverControl solver_control(2000, 1e-6 * system_rhs.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

    // PreconditionBlockDiagonal preconditioner;
    // preconditioner.initialize(system_matrix.block(0, 0),
    //                           pressure_mass.block(1, 1));

    PreconditionBlockTriangularStokes preconditioner;
    preconditioner.initialize(system_matrix.block(0, 0),
                              pressure_mass.block(1, 1),
                              system_matrix.block(1, 0));

    pcout << "Solving the linear system" << std::endl;
    solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
    pcout << "  " << solver_control.last_step() << " GMRES iterations"
          << std::endl;

    solution = solution_owned;
}

void Stokes::output()
{
    pcout << "===============================================" << std::endl;

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
    const Vector<double> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    // Generate patches
    data_out.build_patches();

    std::string numProcessors = std::to_string(mpi_size);
    numProcessors += (mpi_size == 1) ? "_processor" : "_processors";

    // Set output file name
    const std::string output_file_name = "output-Stokes-" + numProcessors;

    std::string output_dir = get_output_directory();

    data_out.write_vtu_with_pvtu_record(output_dir,
                                        output_file_name,
                                        0,
                                        MPI_COMM_WORLD);
}

std::string Stokes::get_output_directory()
{
    namespace fs = std::filesystem;

    if (!fs::exists("./outputs"))
    {
        fs::create_directory("./outputs");
    }

    if (!fs::exists("./outputs/steadyNavierStokes"))
    {
        fs::create_directory("./outputs/steadyNavierStokes");
    }

    std::string sub_dir_name = "outputs_reynolds_" + std::to_string(static_cast<int>(reynolds_number));
    fs::path sub_dir_path = "./outputs/steadyNavierStokes/" + sub_dir_name + "/";
    if (!fs::exists(sub_dir_path))
    {
        fs::create_directory(sub_dir_path);
    }
    fs::path sub_sub_dir_path = sub_dir_path.string() + "Stokes/";

    if (fs::exists(sub_sub_dir_path))
    {
        for (const auto &entry : fs::directory_iterator(sub_sub_dir_path))
        {
            fs::remove_all(entry.path());
        }
    }
    else
    {
        fs::create_directory(sub_sub_dir_path);
    }

    return sub_sub_dir_path.string();
}
