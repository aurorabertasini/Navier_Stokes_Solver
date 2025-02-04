#ifndef NAVIERSTOKES_HPP
#define NAVIERSTOKES_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// ---------------------------------------------------------------------

#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>


using namespace dealii;

// to write function in space use p[0], p[1], p[2], get_time() for x, y, z, t respectively
// to write exponential function use pow(base, exp), for example pow(M_E, 2*p[0]) for exp(2x)
// to write sin function use std::sin, for example std::sin(M_PI*p[0]) for sin(pi*x)
template <unsigned int dim>
class MonolithicNavierStokes
{
public:
    // 2. Function for the transport coefficient.
    class ForcingTerm : public Function<dim>
    {
    public:
        virtual void
        vector_value(const Point<dim> & /*p*/,
                     Vector<double> &values) const override
        {
            for (unsigned int i = 0; i < dim; ++i)
                values[i] = 0.0;
        }

        virtual double
        value(const Point<dim> & /*p*/,
              const unsigned int /*component*/ = 0) const override
        {
            return 0.0;
        }

    protected:
    };

    class InletVelocity : public Function<dim>
    {
    public:
        InletVelocity(const double H)
            : Function<dim>(dim)
        {
            this->H = H;
            if constexpr (dim == 2)
                this->uM = 1.5;
            else
                this->uM = 2.25;
        }

        virtual void
        vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            if constexpr (dim == 2)
                values[0] = 4.0 * uM * p[1] * (H - p[1]) / (H * H);
            else
                values[0] = 16.0 * uM * p[1] * (H - p[1]) * p[2] * (H - p[2]) / (H * H * H * H);
            for (unsigned int i = 1; i < dim; ++i)
                values[i] = 0.0;
        }

        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            if (component == 0)
                if constexpr (dim == 2)
                    return 4.0 * uM * p[1] * (H - p[1]) / (H * H);
                else
                    return 16.0 * uM * p[1] * (H - p[1]) * p[2] * (H - p[2]) / (H * H * H * H);
            else
                return 0.0;
        }

        double get_u_max() const
        {
            return uM;
        }

    protected:
        double uM;
        double H;
    };

    class Function_Neumann : public Function<dim>
    {
    public:
        Function_Neumann()
            : Function<dim>(dim + 1)
        {
        }

        virtual void
        vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            values[0] = 0.0;

            for (unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
        }

        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            if (component == 0)
                return 0.0;
            else
                return 0.0;
        }
    };

    // Function for the initial condition.
    class Function_u0 : public Function<dim>
    {
    public:
        virtual void
        vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            values[0] = 0.0;

            for (unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
        }

        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            if (component == 0)
                return 0.0;
            else
                return 0.0;
        }
    };


/*****************************************************************/
/*****************************************************************/
/*         PRECONDITIONERS OPTIONS                               */
/*****************************************************************/
/*****************************************************************/


// Abstract class for block preconditioners

class BlockPrecondition
{
public:
  virtual ~BlockPrecondition() = default;
  virtual void vmult(TrilinosWrappers::MPI::BlockVector &dst,
                     const TrilinosWrappers::MPI::BlockVector &src) const = 0;
};

/**
 * Block-diagonal preconditioner for Navier–Stokes or Stokes:
 * Each diagonal block (velocity, pressure) uses an inner preconditioner
 * that can be ILU or AMG. We provide two “initialize” methods.
 */

// Trivial Preconditioner
  class PreconditionIdentity : public BlockPrecondition
  {
  public:
    void vmult(TrilinosWrappers::MPI::BlockVector       &dst,
               const TrilinosWrappers::MPI::BlockVector &src) const
    {
      dst = src;
    }
  };
  

class PreconditionBlockDiagonal : public BlockPrecondition
{
public:
  /**
   * ILU-based initialization for both velocity and pressure blocks.
   */
//   // Block-diagonal preconditioner.
// Adapted from the one proposed for the Stokes problem in laboratory 9.
  void initialize_inner_preconditioner(
        std::shared_ptr<TrilinosWrappers::PreconditionBase> &preconditioner,
        const TrilinosWrappers::SparseMatrix &matrix, bool use_ilu) {
    if (use_ilu) {
        std::shared_ptr<TrilinosWrappers::PreconditionILU> actual_preconditioner =
            std::make_shared<TrilinosWrappers::PreconditionILU>();
        actual_preconditioner->initialize(matrix);
        preconditioner = actual_preconditioner;
    } else {
        std::shared_ptr<TrilinosWrappers::PreconditionAMG> actual_preconditioner =
            std::make_shared<TrilinosWrappers::PreconditionAMG>();
        actual_preconditioner->initialize(matrix);
        preconditioner = actual_preconditioner;
    }
    }


  // Initialize the preconditioner.
  void initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                  const TrilinosWrappers::SparseMatrix &pressure_mass_,
                  const unsigned int &maxit_, const double &tol_,
                  const bool &use_ilu)
  {
  maxit = maxit_;
  tol = tol_;
  // Save a reference to the input matrices.
  velocity_stiffness = &velocity_stiffness_;
  pressure_mass = &pressure_mass_;

  // Initialize the preconditioners.
  initialize_inner_preconditioner(preconditioner_velocity, *velocity_stiffness,
                                  use_ilu);
  initialize_inner_preconditioner(preconditioner_pressure, *pressure_mass,
                                  use_ilu);
}

  // Application of the preconditioner.
  void vmult(TrilinosWrappers::MPI::BlockVector &dst,
             const TrilinosWrappers::MPI::BlockVector &src) const override

  {
  // Solve the top-left block.
  SolverControl solver_control_velocity(maxit, tol * src.block(0).l2_norm());
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_velocity(
      solver_control_velocity);
  solver_gmres_velocity.solve(*velocity_stiffness, dst.block(0), src.block(0),
                              *preconditioner_velocity);

  // Solve the bottom-right block.
  SolverControl solver_control_pressure(maxit, tol * src.block(1).l2_norm());
  SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
      solver_control_pressure);
  solver_cg_pressure.solve(*pressure_mass, dst.block(1), src.block(1),
                           *preconditioner_pressure);
  }


 private:
  // Velocity stiffness matrix.
  const TrilinosWrappers::SparseMatrix *velocity_stiffness;

  // Preconditioner used for the velocity block.
  std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_velocity;

  // Pressure mass matrix.
  const TrilinosWrappers::SparseMatrix *pressure_mass;

  // Preconditioner used for the pressure block.
  std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_pressure;

  // Maximum number of iterations for the inner solvers.
  unsigned int maxit;

  // Tolerance for the inner solvers.
  double tol;
};


// SIMPLE preconditioner.
class PreconditionSIMPLE : public BlockPrecondition {
 public:

    void initialize_inner_preconditioner(
        std::shared_ptr<TrilinosWrappers::PreconditionBase> &preconditioner,
        const TrilinosWrappers::SparseMatrix &matrix, bool use_ilu) {
    if (use_ilu) {
        std::shared_ptr<TrilinosWrappers::PreconditionILU> actual_preconditioner =
            std::make_shared<TrilinosWrappers::PreconditionILU>();
        actual_preconditioner->initialize(matrix);
        preconditioner = actual_preconditioner;
    } else {
        std::shared_ptr<TrilinosWrappers::PreconditionAMG> actual_preconditioner =
            std::make_shared<TrilinosWrappers::PreconditionAMG>();
        actual_preconditioner->initialize(matrix);
        preconditioner = actual_preconditioner;
    }
    }

  // Initialize the preconditioner.
  void initialize(const TrilinosWrappers::SparseMatrix &F_matrix_,
                  const TrilinosWrappers::SparseMatrix &negB_matrix_,
                  const TrilinosWrappers::SparseMatrix &Bt_matrix_,
                  const TrilinosWrappers::MPI::BlockVector &vec,
                  const double &alpha_, const unsigned int &maxit_,
                  const double &tol_, const bool &use_ilu)
    {
  alpha = alpha_;
  maxit = maxit_;
  tol = tol_;
  // Save a reference to the input matrices.
  F_matrix = &F_matrix_;
  negB_matrix = &negB_matrix_;
  Bt_matrix = &Bt_matrix_;

  // Save the negated inverse diagonal of F.
  negDinv_vector.reinit(vec.block(0));
  for (unsigned int index : negDinv_vector.locally_owned_elements()) {
    negDinv_vector[index] = -1.0 / F_matrix->diag_element(index);
  }

  // Create the matrix S.
  negB_matrix->mmult(S_matrix, *Bt_matrix, negDinv_vector);

  // Initialize the preconditioners.
  initialize_inner_preconditioner(preconditioner_F, *F_matrix, use_ilu);
  initialize_inner_preconditioner(preconditioner_S, S_matrix, use_ilu);
}

  // Application of the preconditioner.
  void vmult(
    TrilinosWrappers::MPI::BlockVector &dst,
    const TrilinosWrappers::MPI::BlockVector &src) const {
  tmp.reinit(src);
  // Step 1: solve [F 0; B -S]sol1 = src.
  // Step 1.1: solve F*sol1_u = src_u.
  SolverControl solver_control_F(maxit, tol * src.block(0).l2_norm());
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver_F(solver_control_F);
  solver_F.solve(*F_matrix, tmp.block(0), src.block(0), *preconditioner_F);
  // Step 1.2: solve S*sol1_p = B*sol1_u - src_p.
  Bt_matrix->Tvmult(tmp.block(1), tmp.block(0));
  tmp.block(1) -= src.block(1);
  SolverControl solver_control_S(maxit, tol * tmp.block(1).l2_norm());
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
  solver_S.solve(S_matrix, dst.block(1), tmp.block(1), *preconditioner_S);

  // Step 2: solve [I D^-1*B^T; 0 alpha*I]dst = sol1.
  // Step 2.1: solve alpha*I*dst_p = sol1_p.
  dst.block(1) /= alpha;
  // Step 2.2: solve dst_u = sol1_u - D^-1*B^T*dst_p.
  dst.block(0) = tmp.block(0);
  Bt_matrix->vmult(tmp.block(0), dst.block(1));
  tmp.block(0).scale(negDinv_vector);
  dst.block(0) += tmp.block(0);
}


 private:
  // Damping parameter (must be in (0,1]).
  double alpha;

  // Matrix F (top left block of the system matrix).
  const TrilinosWrappers::SparseMatrix *F_matrix;

  // Matrix -B (bottom left block of the system matrix).
  const TrilinosWrappers::SparseMatrix *negB_matrix;

  // Matrix B^T (top right block of the system matrix).
  const TrilinosWrappers::SparseMatrix *Bt_matrix;

  // Matrix -D^-1, negative inverse diagonal of F.
  TrilinosWrappers::MPI::Vector negDinv_vector;

  // Matrix S := B*D^-1*B^T.
  TrilinosWrappers::SparseMatrix S_matrix;

  // Preconditioner used for the block multiplied by F.
  std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_F;

  // Preconditioner used for the block multiplied by S.
  std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_S;

  // Temporary vector.
  mutable TrilinosWrappers::MPI::BlockVector tmp;

  // Maximum number of iterations for the inner solvers.
  unsigned int maxit;

  // Tolerance for the inner solvers.
  double tol;
};


// aSIMPLE preconditioner.
class PreconditionaSIMPLE : public BlockPrecondition {
 public:

 void initialize_inner_preconditioner(
        std::shared_ptr<TrilinosWrappers::PreconditionBase> &preconditioner,
        const TrilinosWrappers::SparseMatrix &matrix, bool use_ilu) {
    if (use_ilu) {
        std::shared_ptr<TrilinosWrappers::PreconditionILU> actual_preconditioner =
            std::make_shared<TrilinosWrappers::PreconditionILU>();
        actual_preconditioner->initialize(matrix);
        preconditioner = actual_preconditioner;
    } else {
        std::shared_ptr<TrilinosWrappers::PreconditionAMG> actual_preconditioner =
            std::make_shared<TrilinosWrappers::PreconditionAMG>();
        actual_preconditioner->initialize(matrix);
        preconditioner = actual_preconditioner;
    }
    }

void initialize(
    const TrilinosWrappers::SparseMatrix &F_matrix_,
    const TrilinosWrappers::SparseMatrix &negB_matrix_,
    const TrilinosWrappers::SparseMatrix &Bt_matrix_,
    const TrilinosWrappers::MPI::BlockVector &vec, const double &alpha_,
    const bool &use_inner_solver_, const unsigned int &maxit_,
    const double &tol_, const bool &use_ilu) 
{
  alpha = alpha_;
  use_inner_solver = use_inner_solver_;
  maxit = maxit_;
  tol = tol_;
  // Save a reference to the input matrices.
  F_matrix = &F_matrix_;
  negB_matrix = &negB_matrix_;
  Bt_matrix = &Bt_matrix_;

  // Save the diagonal and inverse diagonal of F.
  D_vector.reinit(vec.block(0));
  Dinv_vector.reinit(vec.block(0));
  for (unsigned int index : D_vector.locally_owned_elements()) {
    const double value = F_matrix->diag_element(index);
    D_vector[index] = value;
    Dinv_vector[index] = 1.0 / value;
  }

  // Create the matrix -S.
  negB_matrix->mmult(negS_matrix, *Bt_matrix, Dinv_vector);

  // Initialize the preconditioners.
  initialize_inner_preconditioner(preconditioner_F, *F_matrix, use_ilu);
  initialize_inner_preconditioner(preconditioner_S, negS_matrix, use_ilu);
}


  // Application of the preconditioner.
  void vmult(TrilinosWrappers::MPI::BlockVector &dst,
             const TrilinosWrappers::MPI::BlockVector &src) const 
  {
  tmp.reinit(src);
  // Step 1: multiply src by [F^-1 0; 0 I].
  if (use_inner_solver) {
    SolverControl solver_control_F(maxit, tol * src.block(0).l2_norm());
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver_F(solver_control_F);
    solver_F.solve(*F_matrix, dst.block(0), src.block(0), *preconditioner_F);
  } else {
    preconditioner_F->vmult(dst.block(0), src.block(0));
  }
  tmp.block(1) = src.block(1);
  // Step 2: multiply the result by [I 0; -B I].
  negB_matrix->vmult_add(tmp.block(1), dst.block(0));
  // Step 3: multiply the result by [I 0; 0 -S^-1].
  if (use_inner_solver) {
    SolverControl solver_control_S(maxit, tol * tmp.block(1).l2_norm());
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
    solver_S.solve(negS_matrix, dst.block(1), tmp.block(1), *preconditioner_S);
  } else {
    preconditioner_S->vmult(dst.block(1), tmp.block(1));
  }
  // Step 4: multiply the result by [D 0; 0 I/alpha].
  dst.block(0).scale(D_vector);
  dst.block(1) /= alpha;
  // Step 5: multiply the result by [I -B^T; 0 I].
  Bt_matrix->vmult(tmp.block(0), dst.block(1));
  dst.block(0) -= tmp.block(0);
  // Step 6: multiply the result by [D^-1 0; 0 I].
  dst.block(0).scale(Dinv_vector);
}

 private:
  // Damping parameter (must be in (0,1]).
  double alpha;

  // Matrix F (top left block of the system matrix).
  const TrilinosWrappers::SparseMatrix *F_matrix;

  // Matrix -B (bottom left block of the system matrix).
  const TrilinosWrappers::SparseMatrix *negB_matrix;

  // Matrix B^T (top right block of the system matrix).
  const TrilinosWrappers::SparseMatrix *Bt_matrix;

  // Matrix D, diagonal of F.
  TrilinosWrappers::MPI::Vector D_vector;

  // Matrix D^-1.
  TrilinosWrappers::MPI::Vector Dinv_vector;

  // Matrix -S := -B*D^-1*B^T.
  TrilinosWrappers::SparseMatrix negS_matrix;

  // Preconditioner used for the block multiplied by F.
  std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_F;

  // Preconditioner used for the block multiplied by S.
  std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_S;

  // Temporary vector.
  mutable TrilinosWrappers::MPI::BlockVector tmp;

  // Whether to use inner solvers.
  bool use_inner_solver;

  // Maximum number of iterations for the inner solvers.
  unsigned int maxit;

  // Tolerance for the inner solvers.
  double tol;
};


// Yosida preconditioner.
class PreconditionYosida : public BlockPrecondition {
 public:

 void initialize_inner_preconditioner(
        std::shared_ptr<TrilinosWrappers::PreconditionBase> &preconditioner,
        const TrilinosWrappers::SparseMatrix &matrix, bool use_ilu) {
    if (use_ilu) {
        std::shared_ptr<TrilinosWrappers::PreconditionILU> actual_preconditioner =
            std::make_shared<TrilinosWrappers::PreconditionILU>();
        actual_preconditioner->initialize(matrix);
        preconditioner = actual_preconditioner;
    } else {
        std::shared_ptr<TrilinosWrappers::PreconditionAMG> actual_preconditioner =
            std::make_shared<TrilinosWrappers::PreconditionAMG>();
        actual_preconditioner->initialize(matrix);
        preconditioner = actual_preconditioner;
    }
    }

  // Initialize the preconditioner.
  void initialize(const TrilinosWrappers::SparseMatrix &F_matrix_,
                  const TrilinosWrappers::SparseMatrix &negB_matrix_,
                  const TrilinosWrappers::SparseMatrix &Bt_matrix_,
                  const TrilinosWrappers::SparseMatrix &M_dt_matrix_,
                  const TrilinosWrappers::MPI::BlockVector &vec,
                  const unsigned int &maxit_, const double &tol_,
                  const bool &use_ilu)
    {
  maxit = maxit_;
  tol = tol_;
  // Save a reference to the input matrices.
  F_matrix = &F_matrix_;
  negB_matrix = &negB_matrix_;
  Bt_matrix = &Bt_matrix_;

  // Save the inverse diagonal of M_dt.
  Dinv_vector.reinit(vec.block(0));
  for (unsigned int index : Dinv_vector.locally_owned_elements()) {
    Dinv_vector[index] = 1.0 / M_dt_matrix_.diag_element(index);
  }

  // Create the matrix -S.
  negB_matrix->mmult(negS_matrix, *Bt_matrix, Dinv_vector);

  // Initialize the preconditioners.
  initialize_inner_preconditioner(preconditioner_F, *F_matrix, use_ilu);
  initialize_inner_preconditioner(preconditioner_S, negS_matrix, use_ilu);
}

  // Application of the preconditioner.
  void vmult(TrilinosWrappers::MPI::BlockVector &dst,
             const TrilinosWrappers::MPI::BlockVector &src) const
    {
  tmp.reinit(src);
  // Step 1: solve [F 0; B -S]sol1 = src.
  // Step 1.1: solve F*sol1_u = src_u.
  tmp.block(0) = dst.block(0);
  SolverControl solver_control_F(maxit, tol * src.block(0).l2_norm());
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver_F(solver_control_F);
  solver_F.solve(*F_matrix, tmp.block(0), src.block(0), *preconditioner_F);
  // Step 1.2: solve -S*sol1_p = -B*sol1_u + src_p.
  tmp.block(1) = src.block(1);
  negB_matrix->vmult_add(tmp.block(1), tmp.block(0));
  SolverControl solver_control_S(maxit, tol * tmp.block(1).l2_norm());
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
  solver_S.solve(negS_matrix, dst.block(1), tmp.block(1), *preconditioner_S);

  // Step 2: solve [I F^-1*B^T; 0 I]dst = sol1.
  tmp_2 = src.block(0);
  dst.block(0) = tmp.block(0);
  Bt_matrix->vmult(tmp.block(0), dst.block(1));
  SolverControl solver_control_F2(maxit, tol * tmp.block(0).l2_norm());
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_F2(solver_control_F);
  solver_gmres_F2.solve(*F_matrix, tmp_2, tmp.block(0), *preconditioner_F);
  dst.block(0) -= tmp_2;
 }


 private:
  // Matrix F (top left block of the system matrix).
  const TrilinosWrappers::SparseMatrix *F_matrix;

  // Matrix -B (bottom left block of the system matrix).
  const TrilinosWrappers::SparseMatrix *negB_matrix;

  // Matrix B^T (top right block of the system matrix).
  const TrilinosWrappers::SparseMatrix *Bt_matrix;

  // Matrix D^-1, inverse diagonal of M/deltat.
  TrilinosWrappers::MPI::Vector Dinv_vector;

  // Matrix -S := -B*D^-1*B^T.
  TrilinosWrappers::SparseMatrix negS_matrix;

  // Preconditioner used for the block multiplied by F.
  std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_F;

  // Preconditioner used for the block multiplied by S.
  std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_S;

  // Temporary vectors.
  mutable TrilinosWrappers::MPI::BlockVector tmp;
  mutable TrilinosWrappers::MPI::Vector tmp_2;

  // Maximum number of iterations for the inner solvers.
  unsigned int maxit;

  // Tolerance for the inner solvers.
  double tol;
};


/*****************************************************************/
/*****************************************************************/
/*         END PRECONDITIONERS SECTION                           */
/*****************************************************************/
/*****************************************************************/

    // parameter as constructor arguments.
    MonolithicNavierStokes(
        const std::string &mesh_file_name_,
        const unsigned int &degree_velocity_,
        const unsigned int &degree_pressure_,
        const double &T_,
        const double &deltat_,
        const double &re_);
    // Initialization.
    void
    setup();

    // Solve the problem.
    void
    solve(int precond_type, bool use_ilu, double &first_dt_solve_time,
                                        int &first_dt_gmres_iters, double &total_solve_time, double &precond_construct_time);

    void run_with_preconditioners();

    // Compute the error.
    double
    compute_error(const VectorTools::NormType &norm_type);

    std::string
    get_output_directory();

protected:
    // Assemble the matrix without the non-linear term.
    void
    assemble_base_matrix();

    void
    add_convective_term();

    // Assemble the right-hand side of the problem.
    void
    assemble_rhs(const double &time);

    // Solve the problem for one time step.
    void
    solve_time_step(int precond_type, bool use_ilu, double &elapsed_time, int &gmres_iters,
                                                  double &precond_construct_time);

    // Output.
    void
    output(const unsigned int &time_step);

    void
    set_initial_conditions(TrilinosWrappers::MPI::BlockVector solution_stokes_);

    void
    compute_lift_drag();

    // MPI parallel. /////////////////////////////////////////////////////////////

    // extractors for velocity and pressure.
    FEValuesExtractors::Vector velocity;

    FEValuesExtractors::Scalar pressure;

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    // Affine constraints.
    AffineConstraints<double> constraints;

    // diffusion coefficient
    double nu;

    // Reynolds number.
    const double reynolds_number;

    // Height of the channel.
    const double H = 0.41;

    // Inlet velocity.
    InletVelocity inlet_velocity;

    // Cylinder radius.
    const double cylinder_radius = 0.1;

    // // Transport coefficient.
    // Function_t transport;

    // outlet pressure
    const double p_out = 0.0;

    // Forcing term.
    ForcingTerm forcing_term;

    // Function for neuman condition
    Function_Neumann neumann_condition;

    // Function for the initial condition
    Function_u0 initial_condition;

    bool use_inner_solver;

    // // Exact solution.
    // ExactSolution exact_solution;

    // Current time.
    double time;

    // Final time.
    const double T;

    double lift; 

    double drag;

    const double rho = 1.0;

    // Discretization. ///////////////////////////////////////////////////////////

    // Mesh file name.
    const std::string mesh_file_name;

    // Polynomial degree Velocity.
    const unsigned int degree_velocity;

    // Polynomial degree Pressure.
    const unsigned int degree_pressure;

    // Time step.
    const double deltat;

    // Mesh.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // Finite element space.
    std::unique_ptr<FiniteElement<dim>> fe;

    // Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature;

    // Quadrature formula used on boundary lines.
    std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

    // DoF handler.
    DoFHandler<dim> dof_handler;

    // DoFs owned by current process.
    IndexSet locally_owned_dofs;

    std::vector<IndexSet> block_owned_dofs; //!!!!!!!!!!!!!!!!!!

    // DoFs relevant to the current process (including ghost DoFs).
    IndexSet locally_relevant_dofs;

    std::vector<IndexSet> block_relevant_dofs; //!!!!!!!!!!!!!!!!!!

    // Mass matrix M / deltat.
    TrilinosWrappers::BlockSparseMatrix mass_matrix;

    // Global matrix.
    TrilinosWrappers::BlockSparseMatrix system_matrix;

    // velocity mass
    TrilinosWrappers::BlockSparseMatrix velocity_mass;

    // Non_linear term matrix.
    TrilinosWrappers::BlockSparseMatrix convective_matrix;

    // boolean value to choose to reconstruct all the system matrix ad each time step
    bool RECONSTRUCT_SYSTEM_MATRIX = true;

    // Stiffness matrix K.
    TrilinosWrappers::BlockSparseMatrix stiffness_matrix;

    TrilinosWrappers::BlockSparseMatrix pressure_mass;  //!


    TrilinosWrappers::BlockSparseMatrix rhs_matrix;

    // Right-hand side vector in the linear system.
    TrilinosWrappers::MPI::BlockVector system_rhs;

    // System solution (without ghost elements).
    TrilinosWrappers::MPI::BlockVector solution_owned;

    // System solution (including ghost elements).
    TrilinosWrappers::MPI::BlockVector solution;

    ConditionalOStream pcout; // Parallel output stream.

    std::vector<double> vec_drag;
    std::vector<double> vec_lift;
    std::vector<double> vec_drag_coeff;
    std::vector<double> vec_lift_coeff;
};

template <unsigned int dim>
MonolithicNavierStokes<dim>::MonolithicNavierStokes(
        const std::string &mesh_file_name_,
        const unsigned int &degree_velocity_,
        const unsigned int &degree_pressure_,
        const double &T_,
        const double &deltat_,
        const double &re_)
        : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0)
        , mesh_file_name(mesh_file_name_)
        , degree_velocity(degree_velocity_)
        , degree_pressure(degree_pressure_)
        , T(T_)
        , deltat(deltat_)
        , inlet_velocity(H)
        , velocity(0)
        , pressure(dim)
        , mesh(MPI_COMM_WORLD)
        , reynolds_number(re_)
    {
        this->nu = (2./3.) * inlet_velocity.get_u_max() * cylinder_radius / reynolds_number;
    }

#endif