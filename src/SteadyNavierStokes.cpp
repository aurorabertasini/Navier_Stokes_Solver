#include "../include/SteadyNavierStokes.hpp"

// ===============================
// SteadyNavierStokes<dim> methods
// ===============================

template <int dim>
void SteadyNavierStokes<dim>::run_full_problem_pipeline()
{
  this->pcout << "===============================================" << std::endl;
  this->pcout << "Running full pipeline: Stokes -> IncrementalStokes" << std::endl;
  this->pcout << "===============================================" << std::endl;

  // 1) Create a Stokes solver with this object's parameters
  Stokes<dim> stokes_problem(this->mesh_file_name,
                             this->degree_velocity,
                             this->degree_pressure,
                             this->Re);

  // 2) Run the typical steps: setup, assemble, solve, output
  stokes_problem.setup();
  stokes_problem.assemble();
  stokes_problem.solve();
  stokes_problem.output();

  // 3) Retrieve the final solution of the Stokes problem
  TrilinosWrappers::MPI::BlockVector stokes_solution = stokes_problem.get_solution();

  // 4) Create an IncrementalStokes solver from the Stokes problem
  IncrementalStokes<dim> incremental_stokes(stokes_problem);

  // 5) Set the initial condition of the incremental solver to the Stokes solution
  incremental_stokes.set_initial_conditions(stokes_solution);

  // 6) Run the incremental solver steps
  incremental_stokes.setup();
  incremental_stokes.solve();   // Assemble incorporated in solve()
  incremental_stokes.output();

  // 7) Optionally compute lift & drag
  incremental_stokes.compute_lift_drag();
}

template <int dim>
void SteadyNavierStokes<dim>::setup()
{
  this->pcout << "Initializing the mesh" << std::endl;

  // Use a serial Triangulation to read the mesh from file .msh
  Triangulation<dim> mesh_serial;
  {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(this->mesh_file_name);
    AssertThrow(grid_in_file,
                ExcMessage("Could not open mesh file '" + this->mesh_file_name + "'"));

    // Read the .msh file into the serial Triangulation
    grid_in.read_msh(grid_in_file);
  }

  // Partition the mesh among MPI processes
  GridTools::partition_triangulation(this->mpi_size, mesh_serial);

  // Create a parallel Triangulation from the serial one
  const auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      mesh_serial, MPI_COMM_WORLD);
  this->mesh.create_triangulation(construction_data);

  this->pcout << "  Number of elements = "
              << this->mesh.n_global_active_cells() << std::endl;
  this->pcout << "-----------------------------------------------" << std::endl;
}

// Generic methods for the base class, do nothing by default
template <int dim>
void SteadyNavierStokes<dim>::assemble()
{
}

template <int dim>
void SteadyNavierStokes<dim>::solve()
{
}

template <int dim>
void SteadyNavierStokes<dim>::output()
{
}

template <int dim>
std::string SteadyNavierStokes<dim>::get_output_directory()
{
  return "./";  
}

// ===============================
// Stokes<dim> methods
// ===============================

template <int dim>
void Stokes<dim>::setup()
{
  // First, call the base setup() to read/distribute the mesh:
  SteadyNavierStokes<dim>::setup();

  // Now proceed with the "Stokes" specifics:
  this->pcout << "Initializing the finite element space" << std::endl;

  // For velocity and pressure
  const FE_SimplexP<dim> fe_scalar_velocity(this->degree_velocity);
  const FE_SimplexP<dim> fe_scalar_pressure(this->degree_pressure);

  this->fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity, dim,
                                             fe_scalar_pressure,   1);

  this->pcout << "  Velocity degree:           = "
              << fe_scalar_velocity.degree << std::endl;
  this->pcout << "  Pressure degree:           = "
              << fe_scalar_pressure.degree << std::endl;
  this->pcout << "  DoFs per cell              = "
              << this->fe->dofs_per_cell << std::endl;

  this->quadrature = std::make_unique<QGaussSimplex<dim>>(this->fe->degree + 1);
  this->pcout << "  Quadrature points per cell = "
              << this->quadrature->size() << std::endl;

  this->quadrature_face =
      std::make_unique<QGaussSimplex<dim - 1>>(this->fe->degree + 1);
  this->pcout << "  Quadrature points per face = "
              << this->quadrature_face->size() << std::endl;

  this->pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler
  this->pcout << "Initializing the DoF handler" << std::endl;
  this->dof_handler.reinit(this->mesh);
  this->dof_handler.distribute_dofs(*this->fe);

  // Reorder velocity DoFs first, then pressure DoFs
  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1; // pressure is component "dim"
  DoFRenumbering::component_wise(this->dof_handler, block_component);

  // Owned + relevant DoFs
  this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                          this->locally_relevant_dofs);

  // Count how many DoFs are velocity vs. pressure
  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(this->dof_handler, block_component);
  const unsigned int n_u = dofs_per_block[0];
  const unsigned int n_p = dofs_per_block[1];

  this->block_owned_dofs.resize(2);
  this->block_relevant_dofs.resize(2);
  this->block_owned_dofs[0]    = this->locally_owned_dofs.get_view(0, n_u);
  this->block_owned_dofs[1]    = this->locally_owned_dofs.get_view(n_u, n_u + n_p);
  this->block_relevant_dofs[0] = this->locally_relevant_dofs.get_view(0, n_u);
  this->block_relevant_dofs[1] = this->locally_relevant_dofs.get_view(n_u, n_u + n_p);

  this->pcout << "  Number of DoFs: " << std::endl
              << "    velocity = " << n_u << std::endl
              << "    pressure = " << n_p << std::endl
              << "    total    = " << (n_u + n_p) << std::endl;

  this->pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system
  {
    this->pcout << "Initializing the linear system" << std::endl;
    this->pcout << "  Initializing the sparsity pattern" << std::endl;

    // Make the system matrix sparsity
    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
      {
        // For the system matrix, everything except p-p block is "always"
        if (c == dim && d == dim)
          coupling[c][d] = DoFTools::none;
        else
          coupling[c][d] = DoFTools::always;
      }

    TrilinosWrappers::BlockSparsityPattern sparsity(this->block_owned_dofs,
                                                    MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(this->dof_handler,
                                    coupling,
                                    sparsity);
    sparsity.compress();

    // Sparsity pattern for the pressure mass matrix (only p-p block)
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
      {
        if (c == dim && d == dim)
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;
      }

    TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
        this->block_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(this->dof_handler,
                                    coupling,
                                    sparsity_pressure_mass);
    sparsity_pressure_mass.compress();

    this->pcout << "  Initializing the matrices" << std::endl;
    this->system_matrix.reinit(sparsity);
    this->pressure_mass.reinit(sparsity_pressure_mass);

    this->pcout << "  Initializing the system right-hand side" << std::endl;
    this->system_rhs.reinit(this->block_owned_dofs, MPI_COMM_WORLD);

    this->pcout << "  Initializing the solution vector" << std::endl;
    this->solution_owned.reinit(this->block_owned_dofs, MPI_COMM_WORLD);
    this->solution.reinit(this->block_owned_dofs,
                          this->block_relevant_dofs,
                          MPI_COMM_WORLD);
  }
}

template <int dim>
void Stokes<dim>::assemble()
{
  this->pcout << "===============================================" << std::endl;
  this->pcout << "Assembling the system" << std::endl;

  const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
  const unsigned int n_q           = this->quadrature->size();
  const unsigned int n_q_face      = this->quadrature_face->size();

  FEValues<dim> fe_values(*this->fe, *this->quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(*this->fe, *this->quadrature_face,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  this->system_matrix = 0.0;
  this->system_rhs    = 0.0;
  this->pressure_mass = 0.0;

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);
    cell_matrix               = 0.0;
    cell_rhs                  = 0.0;
    cell_pressure_mass_matrix = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      Vector<double> forcing_loc(dim);
      this->forcing_term.vector_value(fe_values.quadrature_point(q), forcing_loc);

      Tensor<1, dim> forcing_tensor;
      for (unsigned int d = 0; d < dim; ++d)
        forcing_tensor[d] = forcing_loc[d];

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          cell_matrix(i, j) += this->nu 
                               * scalar_product(fe_values[velocity].gradient(i, q),
                                                 fe_values[velocity].gradient(j, q))
                               * fe_values.JxW(q);

          cell_matrix(i, j) -= fe_values[velocity].divergence(i, q)
                               * fe_values[pressure].value(j, q)
                               * fe_values.JxW(q);

          cell_matrix(i, j) -= fe_values[velocity].divergence(j, q)
                               * fe_values[pressure].value(i, q)
                               * fe_values.JxW(q);

          cell_pressure_mass_matrix(i, j) +=
                               fe_values[pressure].value(i, q)
                               * fe_values[pressure].value(j, q)
                               / this->nu 
                               * fe_values.JxW(q);
        }

        // Forcing
        cell_rhs(i) += scalar_product(forcing_tensor,
                                      fe_values[velocity].value(i, q))
                       * fe_values.JxW(q);
      }
    }

    // Neumann BC for p_out on boundary_id = 1 (Outlet)
    if (cell->at_boundary())
    {
      for (unsigned int f = 0; f < cell->n_faces(); ++f)
      {
        if ((cell->face(f)->at_boundary()) && (cell->face(f)->boundary_id() == 1))
        {
            fe_face_values.reinit(cell, f);
            for (unsigned int q = 0; q < n_q_face; ++q)
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                // Use base-class protected member "p_out"
                cell_rhs(i) += -(this->p_out)
                                * scalar_product(fe_face_values.normal_vector(q),
                                                fe_face_values[velocity].value(i, q))
                                * fe_face_values.JxW(q);
              }
            } 
        }
      }
    }

    cell->get_dof_indices(dof_indices);
    this->system_matrix.add(dof_indices, cell_matrix);
    this->system_rhs.add(dof_indices, cell_rhs);
    this->pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
  }

  this->system_matrix.compress(VectorOperation::add);
  this->system_rhs.compress(VectorOperation::add);
  this->pressure_mass.compress(VectorOperation::add);

  // Dirichlet boundary conditions
  {
    std::map<types::global_dof_index, double> boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    Functions::ZeroFunction<dim> zero_function(dim + 1);

    // Dirichlet Boundary Conditions
      boundary_functions[0] = &this->inlet_velocity;  // Inlet
      boundary_functions[2] = &zero_function;         // Walls
      boundary_functions[3] = &zero_function;         // Obstacle
    
    if constexpr (dim == 2)
    {
      // Interpolate boundary values into constraints (only velocity components)
      VectorTools::interpolate_boundary_values(
      this->dof_handler,
      boundary_functions,
      boundary_values,
      ComponentMask({true,true, false ,false}));
    }
    else if constexpr (dim == 3)
    {
      // Interpolate boundary values into constraints (only velocity components)
      VectorTools::interpolate_boundary_values(
      this->dof_handler,
      boundary_functions,
      boundary_values,
      ComponentMask({true,true, true,false}));
    }

    // Apply them
    MatrixTools::apply_boundary_values(boundary_values,
                                       this->system_matrix,
                                       this->solution_owned,
                                       this->system_rhs,
                                       false);
                                       
  }
}


template <int dim>
void Stokes<dim>::solve()
{
  this->pcout << "===============================================" << std::endl;

  SolverControl solver_control(2000, 1e-6 * this->system_rhs.l2_norm());
  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

  // Example: use a block triangular preconditioner
  typename SteadyNavierStokes<dim>::PreconditionBlockTriangularStokes preconditioner;
  preconditioner.initialize(this->system_matrix.block(0,0),
                            this->pressure_mass.block(1,1),
                            this->system_matrix.block(1,0));

  this->pcout << "Solving the linear system" << std::endl;
  solver.solve(this->system_matrix, this->solution_owned,
               this->system_rhs, preconditioner);
  this->pcout << "  " << solver_control.last_step() << " GMRES iterations"
              << std::endl;

  // Distribute to the fully relevant solution
  this->solution = this->solution_owned;
}


template <int dim>
void Stokes<dim>::output()
{
  this->pcout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  // velocity = vector, pressure = scalar
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim,
      DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

  // For 2D, we have velocity_x, velocity_y, then pressure
  std::vector<std::string> names;
  for (unsigned int i = 0; i < dim; ++i)
    names.push_back("velocity");
  names.push_back("pressure");

  data_out.add_data_vector(this->dof_handler,
                           this->solution,
                           names,
                           data_component_interpretation);

  // Add partitioning info
  std::vector<unsigned int> partition_int(this->mesh.n_active_cells());
  GridTools::get_subdomain_association(this->mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  std::string numProcessors = std::to_string(this->mpi_size);
  numProcessors += (this->mpi_size == 1) ? "_processor" : "_processors";

  const std::string output_file_name = "output-Stokes-" + numProcessors;
  data_out.write_vtu_with_pvtu_record(this->get_output_directory(),
                                      output_file_name,
                                      0,
                                      MPI_COMM_WORLD);

  this->pcout << "Output written to " << output_file_name << std::endl;
  this->pcout << "===============================================" << std::endl;
}


template <int dim>
std::string Stokes<dim>::get_output_directory()
{
    namespace fs = std::filesystem;

    // 1) Ensure top-level "outputs/" exists
    if (!fs::exists("outputs"))
        fs::create_directory("outputs");

    // 2) Create a subdirectory specific to "SteadyNavierStokes"
    if (!fs::exists("outputs/SteadyNavierStokes"))
        fs::create_directory("outputs/SteadyNavierStokes");

    // 3) Create a subdirectory specific to "Stokes"
    if (!fs::exists("outputs/SteadyNavierStokes/Stokes"))
        fs::create_directory("outputs/SteadyNavierStokes/Stokes");

    // 4) Further subdivide by Reynolds number (or any relevant parameter)
    const std::string sub_dir_name =
        "outputs_reynolds_" + std::to_string(static_cast<int>(this->Re));

    fs::path sub_dir_path = 
        fs::path("outputs/SteadyNavierStokes/Stokes") / sub_dir_name;

    if (!fs::exists(sub_dir_path))
        fs::create_directory(sub_dir_path);

    // Return the absolute string path to the new directory
    return sub_dir_path.string();
}

// ===============================
// IncrementalStokes<dim> methods
// ===============================

template <int dim>
void IncrementalStokes<dim>::setup()
{
  // We do NOT read the mesh again because it was already copied in the construtor.
  // We do want to initialize the FE system, quadrature, etc.

  this->pcout << "Initializing the finite element space" << std::endl;

  const FE_SimplexP<dim> fe_scalar_velocity(this->degree_velocity);
  const FE_SimplexP<dim> fe_scalar_pressure(this->degree_pressure);
  this->fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity, dim,
                                             fe_scalar_pressure,   1);

  this->pcout << "  Velocity degree:           = "
              << fe_scalar_velocity.degree << std::endl;
  this->pcout << "  Pressure degree:           = "
              << fe_scalar_pressure.degree << std::endl;
  this->pcout << "  DoFs per cell              = "
              << this->fe->dofs_per_cell << std::endl;

  this->quadrature = std::make_unique<QGaussSimplex<dim>>(this->fe->degree + 1);
  this->pcout << "  Quadrature points per cell = "
              << this->quadrature->size() << std::endl;

  this->quadrature_face =
      std::make_unique<QGaussSimplex<dim - 1>>(this->fe->degree + 1);
  this->pcout << "  Quadrature points per face = "
              << this->quadrature_face->size() << std::endl;
  this->pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler
  this->pcout << "Initializing the DoF handler" << std::endl;
  this->dof_handler.reinit(this->mesh);
  this->dof_handler.distribute_dofs(*this->fe);

  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(this->dof_handler, block_component);

  this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                          this->locally_relevant_dofs);

  std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(this->dof_handler, block_component);
  const unsigned int n_u = dofs_per_block[0];
  const unsigned int n_p = dofs_per_block[1];

  this->block_owned_dofs.resize(2);
  this->block_relevant_dofs.resize(2);
  this->block_owned_dofs[0]    = this->locally_owned_dofs.get_view(0, n_u);
  this->block_owned_dofs[1]    = this->locally_owned_dofs.get_view(n_u, n_u + n_p);
  this->block_relevant_dofs[0] = this->locally_relevant_dofs.get_view(0, n_u);
  this->block_relevant_dofs[1] = this->locally_relevant_dofs.get_view(n_u, n_u + n_p);

  this->pcout << "  Number of DoFs: " << std::endl
              << "    velocity = " << n_u << std::endl
              << "    pressure = " << n_p << std::endl
              << "    total    = " << (n_u + n_p) << std::endl;
  this->pcout << "-----------------------------------------------" << std::endl;

  // Initialize constraints
  {
    constraints.clear();
    // Define boundary conditions based on dimension
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    Functions::ZeroFunction<dim> zero_function(dim + 1);

    // Dirichlet Boundary Conditions
      boundary_functions[0] = &this->inlet_velocity;  // inlet
      boundary_functions[2] = &zero_function;         // walls
      boundary_functions[3] = &zero_function;         // obstacle

    if constexpr (dim == 2)
    {
      // Interpolate boundary values into constraints (only velocity components)
      VectorTools::interpolate_boundary_values(this->dof_handler,
                                             boundary_functions,
                                             constraints,
                                             ComponentMask({true,true,false,false}));
    }
    else if constexpr (dim == 3)
    {
      // Interpolate boundary values into constraints (only velocity components)
      VectorTools::interpolate_boundary_values(this->dof_handler,
                                             boundary_functions,
                                             constraints,
                                             ComponentMask({true,true,true,false}));
    }

    constraints.close();
  }

  // Initialize system
  this->pcout << "Initializing the linear system" << std::endl;
  this->pcout << "Initializing the sparsity pattern" << std::endl;

  Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
  for (unsigned int c = 0; c < dim + 1; ++c)
    for (unsigned int d = 0; d < dim + 1; ++d)
      coupling[c][d] = DoFTools::always; // we assume all couplings

  TrilinosWrappers::BlockSparsityPattern sparsity(this->block_owned_dofs,
                                                  MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(this->dof_handler,
                                  coupling,
                                  sparsity,
                                  constraints,
                                  false);
  sparsity.compress();

  // Pressure mass matrix pattern (only p-p block)
  for (unsigned int c = 0; c < dim + 1; ++c)
    for (unsigned int d = 0; d < dim + 1; ++d)
    {
      if (c == dim && d == dim)
        coupling[c][d] = DoFTools::always;
      else
        coupling[c][d] = DoFTools::none;
    }

  TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
      this->block_owned_dofs, MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(this->dof_handler,
                                  coupling,
                                  sparsity_pressure_mass,
                                  constraints,
                                  false);
  sparsity_pressure_mass.compress();

  this->system_matrix.reinit(sparsity);
  this->pressure_mass.reinit(sparsity_pressure_mass);

  this->system_rhs.reinit(this->block_owned_dofs, MPI_COMM_WORLD);
  this->solution_owned.reinit(this->block_owned_dofs, MPI_COMM_WORLD);
  this->solution.reinit(this->block_owned_dofs,
                        this->block_relevant_dofs,
                        MPI_COMM_WORLD);
}

template <int dim>
void IncrementalStokes<dim>::assemble()
{
  const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
  const unsigned int n_q           = this->quadrature->size();
  const unsigned int n_q_face      = this->quadrature_face->size();

  FEValues<dim> fe_values(*this->fe, *this->quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(*this->fe, *this->quadrature_face,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     local_rhs(dofs_per_cell);

  // Reinit system
  this->system_matrix = 0.0;
  this->system_rhs    = 0.0;

  // Temporary arrays for old solution evaluations
  std::vector<Tensor<1, dim>> previous_velocity_values(n_q);
  std::vector<Tensor<2, dim>> previous_velocity_gradients(n_q);
  std::vector<double>         previous_pressure_values(n_q);

  // Predefine shape function evaluations
  std::vector<double>         div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double>         phi_p(dofs_per_cell);

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned()) 
      continue;

    fe_values.reinit(cell);

    local_matrix = 0.0;
    local_rhs    = 0.0;

    // Evaluate old solution at quadrature points
    fe_values[u_k].get_function_values(this->solution_old, previous_velocity_values);
    fe_values[u_k].get_function_gradients(this->solution_old, previous_velocity_gradients);
    fe_values[p_k].get_function_values(this->solution_old, previous_pressure_values);

    for (unsigned int q = 0; q < n_q; ++q)
    {
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        div_phi_u[k]  = fe_values[u_k].divergence(k, q);
        grad_phi_u[k] = fe_values[u_k].gradient(k, q);
        phi_u[k]      = fe_values[u_k].value(k, q);
        phi_p[k]      = fe_values[p_k].value(k, q);
      }

      // Build local contributions
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          local_matrix(i, j) += this->nu 
                                * scalar_product(grad_phi_u[i], grad_phi_u[j]) 
                                * fe_values.JxW(q);

          local_matrix(i, j) += phi_u[j] * transpose(previous_velocity_gradients[q]) 
                                * phi_u[i]
                                * fe_values.JxW(q);

          local_matrix(i, j) += previous_velocity_values[q] 
                                * transpose(grad_phi_u[j]) 
                                * phi_u[i]
                                * fe_values.JxW(q);

          local_matrix(i, j) -= phi_p[j] * div_phi_u[i] * fe_values.JxW(q);
          local_matrix(i, j) -= phi_p[i] * div_phi_u[j] * fe_values.JxW(q);
        }

        local_rhs[i] += previous_velocity_values[q] 
                        * transpose(previous_velocity_gradients[q])
                        * phi_u[i] 
                        * fe_values.JxW(q);
      }
    }

     // Neumann boundary condition for p_out on boundary_id = 2 
        if (cell->at_boundary())
        {
            for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
                if (cell->face(f)->at_boundary() && (cell->face(f)->boundary_id() == 2))
                {
                  fe_face_values.reinit(cell, f);
                  for (unsigned int q = 0; q < n_q_face; ++q)
                  {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                          local_rhs[i] += -(this->p_out)
                                          * scalar_product(fe_face_values.normal_vector(q),
                                                          fe_face_values[u_k].value(i, q))
                                          * fe_face_values.JxW(q);
                      }
                  } 
                }
            }
        }

    cell->get_dof_indices(dof_indices);

    constraints.distribute_local_to_global(local_matrix,
                                           local_rhs,
                                           dof_indices,
                                           this->system_matrix,
                                           this->system_rhs);
  }
  
  this->system_matrix.compress(VectorOperation::add);
  this->system_rhs.compress(VectorOperation::add);
}


template <int dim>
void IncrementalStokes<dim>::solve()
{
  for (iter = 0; iter < maxIter; ++iter)
  {
    // Each iteration re-assembles with the updated solution
    this->assemble();
    this->pcout << "RHS Norm Value at iteration " << this->iter
                << " => " << this->system_rhs.l2_norm() << std::endl;

    double update_norm = update_tol + 1.0;

    if (update_norm < update_tol)
      break;
    else
    {
      SolverControl solver_control(2'000'000, 1e-4);
      SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

      // Very simple preconditioner: Identity
      typename SteadyNavierStokes<dim>::PreconditionIdentity preconditioner;
      constraints.set_zero(this->solution_owned); // impose homogeneous constraints on guess

      solver.solve(this->system_matrix,
                   this->solution_owned,
                   this->system_rhs,
                   preconditioner);

      constraints.distribute(this->solution_owned);

      this->pcout << "  " << solver_control.last_step()
                  << " GMRES iterations" << std::endl;

      this->solution = this->solution_owned;

      // Evaluate update = (solution - solution_old)
      this->new_res.reinit(this->solution);
      this->new_res = this->solution;
      this->new_res.sadd(1.0, -1.0, this->solution_old);

      // Compute L2 norm
      double local_sum = 0.0;
      for (unsigned int k = 0; k < this->new_res.size(); ++k)
        local_sum += this->new_res(k) * this->new_res(k);

      double global_sum = 0.0;
      MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      update_norm = std::sqrt(global_sum);

      if (this->mpi_rank == 0)
        std::cout << "L2 norm of the update = " << update_norm << std::endl;

      this->solution_old = this->solution;

      if (this->mpi_rank == 0)
        std::cout << "Iteration " << iter << " completed." << std::endl;

      if (update_norm < update_tol)
        break;
    }
  }
}

template <int dim>
void IncrementalStokes<dim>::output()
{
  this->pcout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim,
      DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

  std::vector<std::string> names;
  for (unsigned int i = 0; i < dim; ++i)
    names.push_back("velocity");
  names.push_back("pressure");

  data_out.add_data_vector(this->dof_handler,
                           this->solution,
                           names,
                           data_component_interpretation);

  // Partition info
  std::vector<unsigned int> partition_int(this->mesh.n_active_cells());
  GridTools::get_subdomain_association(this->mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  std::string numProcessors = std::to_string(this->mpi_size);
  numProcessors += (this->mpi_size == 1) ? "_processor" : "_processors";

  const std::string output_file_name = "output-IncrementalStokes-" + numProcessors;
  data_out.write_vtu_with_pvtu_record(this->get_output_directory(),
                                      output_file_name,
                                      0,
                                      MPI_COMM_WORLD);

  this->pcout << "Output written to " << output_file_name << std::endl;
  this->pcout << "===============================================" << std::endl;
}


template <int dim>
void IncrementalStokes<dim>::compute_lift_drag()
{
    // Define quadrature for faces
    QGauss<dim - 1> face_quadrature_formula(3);
    const unsigned int n_q_points = face_quadrature_formula.size();

    // Define FE extractors for velocity and pressure
    FEValuesExtractors::Vector velocities(0);
    FEValuesExtractors::Scalar pressure(dim);

    // Containers to store evaluated values
    std::vector<double>         pressure_values(n_q_points);
    std::vector<Tensor<2, dim>> velocity_gradients(n_q_points);

    // Initialize FE face values
    FEFaceValues<dim> fe_face_values(*this->fe,
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                     update_gradients | update_JxW_values |
                                     update_normal_vectors);

    // Initialize tensors for calculations
    Tensor<1, dim> normal_vector;
    Tensor<2, dim> fluid_stress;
    Tensor<2, dim> fluid_pressure;
    Tensor<1, dim> forces;

    // Initialize drag and lift accumulators
    double local_drag = 0.0;
    double local_lift = 0.0;

    // Iterate over all cells
    for (const auto &cell : this->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;
            
        if (!cell->at_boundary())
            continue;

        for (unsigned int f = 0; f < cell->n_faces(); ++f)
        {
            if (!cell->face(f)->at_boundary())
                continue;

            unsigned int boundary_id = cell->face(f)->boundary_id();
            bool is_stress_boundary = false;

            // Determine if current face is where stress should be evaluated           
            if (boundary_id == 3) // Obstacle 
                is_stress_boundary = true;

            if (!is_stress_boundary)
                continue;

            // Reinitialize FE face values for the current face
            fe_face_values.reinit(cell, f);

            // Retrieve velocity gradients and pressure values on the face
            fe_face_values[velocities].get_function_gradients(this->solution, velocity_gradients);
            fe_face_values[pressure].get_function_values(this->solution, pressure_values);

            // Iterate over quadrature points on the face
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                // Get the normal vector (ensure consistent orientation)
                normal_vector = -fe_face_values.normal_vector(q);

                // Compute fluid pressure tensor (p * I)
                fluid_pressure = 0;
                for (unsigned int d = 0; d < dim; ++d)
                    fluid_pressure[d][d] = pressure_values[q];

                // Compute fluid stress tensor (nu * grad(U) - pI)
                fluid_stress = this->nu * velocity_gradients[q] - fluid_pressure;

                // Compute forces: stress tensor contracted with normal vector and scaled by JxW
                forces = fluid_stress * normal_vector * fe_face_values.JxW(q);

                // Accumulate drag and lift using the scaling factor
                local_drag += this->scaling_factor * forces[0];
                local_lift += this->scaling_factor * forces[1];
            }
        }
    }

    // Print local partial results for debugging
    std::cout << "Rank " << this->mpi_rank << ": Local Drag = " << local_drag
              << ", Local Lift = " << local_lift << std::endl;

    // Reduce lift and drag across all processes to rank 0
    double total_lift = 0.0;
    double total_drag = 0.0;

    MPI_Reduce(&local_lift, &total_lift, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_drag, &total_drag, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Define points of interest for pressure difference
    std::vector<Point<dim>> points_of_interest;
    if constexpr (dim == 2)
    {
        points_of_interest.emplace_back(0.15, 0.20);
        points_of_interest.emplace_back(0.25, 0.20);
    }
    else if constexpr (dim == 3)
    {
        points_of_interest.emplace_back(0.45, 0.2, 0.205);
        points_of_interest.emplace_back(0.55, 0.2, 0.205);
    }

    // Containers to store pressure at points
    Vector<double> solution_values1(dim + 1);
    Vector<double> solution_values2(dim + 1);

    bool p1_available = true;
    bool p2_available = true;

    // Attempt to evaluate pressure at p1
    try
    {
        VectorTools::point_value(this->dof_handler, this->solution, points_of_interest[0],
                                 solution_values1);
    }
    catch (const dealii::VectorTools::ExcPointNotAvailableHere &)
    {
        p1_available = false;
    }

    // Attempt to evaluate pressure at p2
    try
    {
        VectorTools::point_value(this->dof_handler, this->solution, points_of_interest[1],
                                 solution_values2);
    }
    catch (const dealii::VectorTools::ExcPointNotAvailableHere &)
    {
        p2_available = false;
    }

    // Initialize pressure variables
    double pres_point1 = 0.0;
    double pres_point2 = 0.0;

    // Assign local pressure values if available
    if (p1_available)
        pres_point1 = solution_values1(dim);
    if (p2_available)
        pres_point2 = solution_values2(dim);

    // Reduce pressure points to rank 0
    double global_pres_point1 = 0.0;
    double global_pres_point2 = 0.0;

    // Assuming only one process has each pressure point, use MPI_MAX to gather the value
    MPI_Reduce(&pres_point1, &global_pres_point1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pres_point2, &global_pres_point2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (this->mpi_rank == 0)
    {
        // Compute pressure difference
        double p_diff = global_pres_point1 - global_pres_point2;
        std::cout << "Pressure difference (P(A) - P(B)) = " << p_diff << std::endl;

        // Write final aggregated results to CSV
        std::string output_path = this->get_output_directory() + "/lift_drag_output.csv";
        std::ofstream output_file(output_path, std::ios::app);
        if (output_file.is_open())
        {
            output_file << total_drag << ", " << total_lift << ", " << p_diff << "\n";
            output_file.close();
            std::cout << "Wrote aggregated drag/lift data to lift_drag_output.csv" << std::endl;
        }
        else
        {
            std::cerr << "Error: Unable to open lift_drag_output.csv for writing." << std::endl;
        }

        // Optionally, print total drag and lift
        std::cout << "Total Drag = " << total_drag << std::endl;
        std::cout << "Total Lift = " << total_lift << std::endl;
    }

    // Ensure all processes have completed the reductions
    MPI_Barrier(MPI_COMM_WORLD);
}

template <int dim>
std::string IncrementalStokes<dim>::get_output_directory()
{
    namespace fs = std::filesystem;

    // 1) Ensure top-level "outputs/" exists
    if (!fs::exists("outputs"))
        fs::create_directory("outputs");

    // 2) Create a subdirectory specific to "SteadyNavierStokes"
    if (!fs::exists("outputs/SteadyNavierStokes"))
        fs::create_directory("outputs/SteadyNavierStokes");
    
    // 3) Create a subdirectory specific to "IncrementalStokes"
    if (!fs::exists("outputs/SteadyNavierStokes/IncrementalStokes"))
        fs::create_directory("outputs/SteadyNavierStokes/IncrementalStokes");

    // 4) Further subdivide by Reynolds number (or any relevant parameter)
    const std::string sub_dir_name = 
        "outputs_reynolds_" + std::to_string(static_cast<int>(this->Re));

    fs::path sub_dir_path = 
        fs::path("outputs/SteadyNavierStokes/IncrementalStokes") / sub_dir_name;

    if (!fs::exists(sub_dir_path))
        fs::create_directory(sub_dir_path);

    // Return the absolute string path to the new directory
    return sub_dir_path.string();
}






// ===============================
// Explicit Instantiations
// ===============================

// If you only want 2D and 3D, do so here:
template class SteadyNavierStokes<2>;
template class Stokes<2>;
template class IncrementalStokes<2>;

template class SteadyNavierStokes<3>;
template class Stokes<3>;
template class IncrementalStokes<3>;