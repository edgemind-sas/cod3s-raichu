//! Dense linear solve: hand-written LU factorisation with partial
//! pivoting.
//!
//! The algebraic blocks need one operation from linear algebra: solve
//! `(I − A)·x = c` for a small dense system, re-factorising only when a
//! coefficient moved. Blocks are small (a handful of equations on the
//! models this engine serves), so a dependency-free solver keeps the
//! four-platform wheel matrix untouched: no linear-algebra crate enters
//! the tree for this.
//!
//! Numerical contract (correctness outranks cleverness):
//!
//! - **Partial pivoting** selects, at step k, the row with the largest
//!   magnitude in column k. Its worst-case growth factor is 2^(n - 1),
//!   so pivoting alone does not guarantee a small backward error for all
//!   matrices. The well-conditioned test systems require a relative
//!   residual below 1e-12 through order 50.
//! - The **relative pivot test** refuses a factorisation whose pivot
//!   magnitude falls at or below [`DEFAULT_PIVOT_TOLERANCE`] times the
//!   magnitude of the largest entry of the *original* matrix. Relative,
//!   because a system scaled by 1e-30 is as solvable as one scaled by
//!   1e30; against the original matrix, because the intermediate
//!   entries of a nearly-singular system can grow, and a pivot large
//!   next to its own column but small next to the matrix is precisely
//!   the case the test exists to catch.

use thiserror::Error;

/// Default relative floor under which a pivot counts as zero, as a
/// fraction of the largest entry magnitude of the original matrix.
/// This dimensionless threshold is a numerical-singularity policy, not
/// a condition-number estimate or a guarantee on solution accuracy.
pub const DEFAULT_PIVOT_TOLERANCE: f64 = 1e-12;

/// A dense matrix could not be factorised: at some elimination step
/// every remaining entry of the pivot column is (relatively) zero, so
/// the matrix is singular or numerically singular.
#[derive(Debug, Clone, PartialEq, Error)]
#[error(
    "singular matrix: the pivot search at elimination step {step} found no \
     entry above the relative floor (largest magnitude {magnitude:e})"
)]
pub struct SingularMatrix {
    /// Zero-based elimination step at which the search failed.
    pub step: usize,
    /// Largest magnitude the pivot search saw in the column.
    pub magnitude: f64,
}

/// Invalid linear-system input or failed numerical evolution.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum LinearSolveError {
    /// The square matrix dimension cannot be represented as a slice length.
    #[error("matrix dimension {dimension} overflows its square")]
    DimensionOverflow {
        /// Requested order of the square matrix.
        dimension: usize,
    },
    /// A matrix or right-hand side has the wrong number of entries.
    #[error("{input} needs {expected} entries, received {actual}")]
    DimensionMismatch {
        /// Input whose length is invalid.
        input: &'static str,
        /// Required slice length.
        expected: usize,
        /// Actual slice length.
        actual: usize,
    },
    /// The relative pivot tolerance must be finite and strictly positive.
    #[error("relative pivot tolerance must be finite and positive, received {tolerance}")]
    InvalidTolerance {
        /// Rejected relative pivot tolerance.
        tolerance: f64,
    },
    /// An input or intermediate result contains a non-finite entry.
    #[error("non-finite {input} entry at index {index}")]
    NonFinite {
        /// Input or numerical result carrying the entry.
        input: &'static str,
        /// Zero-based entry index (row-major for a matrix).
        index: usize,
    },
    /// No pivot exceeds the relative floor.
    #[error(transparent)]
    Singular(#[from] SingularMatrix),
}

fn check_finite(values: &[f64], input: &'static str) -> Result<(), LinearSolveError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(LinearSolveError::NonFinite { input, index });
    }
    Ok(())
}

/// An `LU` factorisation `P·A = L·U` of a dense square matrix, computed
/// with partial pivoting. Row-major storage; `L` is unit lower
/// triangular and `U` upper triangular, both packed into one `n × n`
/// buffer in the usual in-place way.
#[derive(Debug, Clone)]
pub struct LuFactorization {
    n: usize,
    /// Packed factors: at or above the diagonal `U`, strictly below `L`
    /// (whose unit diagonal is implicit).
    lu: Vec<f64>,
    /// Row swap applied at each elimination step: `pivots[k]` is the row
    /// swapped with `k` at step `k` (LAPACK `ipiv` convention).
    pivots: Vec<usize>,
}

impl LuFactorization {
    /// Factorise `matrix` (`n × n`, row-major, `matrix.len() == n * n`)
    /// under the default relative pivot tolerance.
    ///
    /// # Errors
    /// [`LinearSolveError`] for invalid dimensions, non-finite entries,
    /// invalid tolerance or a singular matrix. An empty 0 by 0 system
    /// is valid and has an empty solution.
    pub fn decompose(matrix: &[f64], n: usize) -> Result<Self, LinearSolveError> {
        Self::decompose_with_tolerance(matrix, n, DEFAULT_PIVOT_TOLERANCE)
    }

    /// Factorise under an explicit relative pivot tolerance, as a
    /// fraction of the largest entry magnitude of `matrix`.
    ///
    /// # Errors
    /// [`LinearSolveError`] for invalid dimensions, non-finite entries,
    /// invalid tolerance or a singular matrix. An empty 0 by 0 system
    /// is valid and has an empty solution.
    pub fn decompose_with_tolerance(
        matrix: &[f64],
        n: usize,
        tolerance: f64,
    ) -> Result<Self, LinearSolveError> {
        let expected = n
            .checked_mul(n)
            .ok_or(LinearSolveError::DimensionOverflow { dimension: n })?;
        if matrix.len() != expected {
            return Err(LinearSolveError::DimensionMismatch {
                input: "matrix",
                expected,
                actual: matrix.len(),
            });
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(LinearSolveError::InvalidTolerance { tolerance });
        }
        check_finite(matrix, "matrix")?;
        let scale = matrix
            .iter()
            .fold(0.0f64, |max, value| max.max(value.abs()));
        let mut lu = matrix.to_vec();
        let mut pivots = vec![0usize; n];
        for k in 0..n {
            // Partial pivoting: the largest remaining entry of column k.
            let (row, magnitude) = (k..n).fold((k, 0.0f64), |best, candidate| {
                let value = lu[candidate * n + k].abs();
                if value > best.1 {
                    (candidate, value)
                } else {
                    best
                }
            });
            // Compare a ratio to avoid underflow or overflow in tolerance * scale.
            if scale == 0.0 || magnitude / scale <= tolerance {
                return Err(SingularMatrix { step: k, magnitude }.into());
            }
            pivots[k] = row;
            if row != k {
                for column in 0..n {
                    lu.swap(k * n + column, row * n + column);
                }
            }
            let pivot = lu[k * n + k];
            for row in (k + 1)..n {
                let factor = lu[row * n + k] / pivot;
                if !factor.is_finite() {
                    return Err(LinearSolveError::NonFinite {
                        input: "factorization",
                        index: row * n + k,
                    });
                }
                lu[row * n + k] = factor;
                if factor != 0.0 {
                    for column in (k + 1)..n {
                        let index = row * n + column;
                        lu[index] -= factor * lu[k * n + column];
                        if !lu[index].is_finite() {
                            return Err(LinearSolveError::NonFinite {
                                input: "factorization",
                                index,
                            });
                        }
                    }
                }
            }
        }
        Ok(LuFactorization { n, lu, pivots })
    }

    /// Solve `A·x = b` for the factorised matrix, writing the solution
    /// over `b` (`b.len() == n`).
    ///
    /// No allocation occurs; the factorisation can be reused for any number
    /// of right-hand sides. Invalid input is rejected before modifying `b`.
    /// A numerical overflow may leave `b` partially updated.
    ///
    /// # Errors
    /// [`LinearSolveError`] for an invalid right-hand side or a non-finite
    /// intermediate solution.
    pub fn solve_into(&self, b: &mut [f64]) -> Result<(), LinearSolveError> {
        if b.len() != self.n {
            return Err(LinearSolveError::DimensionMismatch {
                input: "right-hand side",
                expected: self.n,
                actual: b.len(),
            });
        }
        check_finite(b, "right-hand side")?;
        // Apply the row permutations to the right-hand side.
        for (k, &row) in self.pivots.iter().enumerate() {
            if row != k {
                b.swap(k, row);
            }
        }
        // Forward substitution with the unit lower triangle.
        for row in 0..self.n {
            let mut accumulator = b[row];
            for (column, &value) in b.iter().enumerate().take(row) {
                accumulator -= self.lu[row * self.n + column] * value;
            }
            if !accumulator.is_finite() {
                return Err(LinearSolveError::NonFinite {
                    input: "solution",
                    index: row,
                });
            }
            b[row] = accumulator;
        }
        // Back substitution with the upper triangle.
        for row in (0..self.n).rev() {
            let mut accumulator = b[row];
            for (column, &value) in b.iter().enumerate().skip(row + 1) {
                accumulator -= self.lu[row * self.n + column] * value;
            }
            let value = accumulator / self.lu[row * self.n + row];
            if !value.is_finite() {
                return Err(LinearSolveError::NonFinite {
                    input: "solution",
                    index: row,
                });
            }
            b[row] = value;
        }
        Ok(())
    }

    /// Solve `A·x = b` and return `x` (see [`LuFactorization::solve_into`]).
    ///
    /// # Errors
    /// See [`Self::solve_into`]. This convenience method allocates its output;
    /// use [`Self::solve_into`] with a reusable buffer in simulation loops.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, LinearSolveError> {
        let mut x = b.to_vec();
        self.solve_into(&mut x)?;
        Ok(x)
    }

    /// Order of the system.
    #[must_use]
    pub fn size(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use super::*;

    /// Deterministic 64-bit linear congruential generator (numerical
    /// crust, not cryptography): keeps this crate dependency-free even
    /// at test time.
    struct Lcg(u64);

    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Top 53 bits, mapped to [-1, 1).
            let bits = (self.0 >> 11) as f64;
            bits / (1u64 << 53) as f64 * 2.0 - 1.0
        }
    }

    fn relative_residual(a: &[f64], x: &[f64], b: &[f64], n: usize) -> f64 {
        let mut worst = 0.0f64;
        for row in 0..n {
            let mut product = 0.0;
            for column in 0..n {
                product += a[row * n + column] * x[column];
            }
            worst = worst.max((product - b[row]).abs());
        }
        let scale = b.iter().fold(0.0f64, |max, value| max.max(value.abs()));
        worst / scale.max(1.0)
    }

    #[test]
    fn solves_the_documented_two_by_two() {
        // The coupled shape the blocks produce: x1 = 0.5·x2 + 1 and
        // x2 = 0.25·x1 + 3, so (I − A) = [[1, −0.5], [−0.25, 1]] and the
        // closed form is x = (20/7, 26/7).
        let a = [1.0, -0.5, -0.25, 1.0];
        let lu = LuFactorization::decompose(&a, 2).unwrap();
        let x = lu.solve(&[1.0, 3.0]).unwrap();
        assert!((x[0] - 20.0 / 7.0).abs() < 1e-12, "x = {x:?}");
        assert!((x[1] - 26.0 / 7.0).abs() < 1e-12, "x = {x:?}");
    }

    #[test]
    fn permuted_systems_solve() {
        // A zero on the diagonal forces the pivot: [[0, 1], [1, 0]].
        let lu = LuFactorization::decompose(&[0.0, 1.0, 1.0, 0.0], 2).unwrap();
        assert_eq!(lu.solve(&[3.0, 7.0]).unwrap(), vec![7.0, 3.0]);
    }

    #[test]
    fn random_well_conditioned_systems_solve_to_the_residual_bound() {
        let mut random = Lcg(0x5EED_C0FFEE);
        for n in 1..=50 {
            // Diagonally dominant draw: well conditioned by construction.
            let mut a = vec![0.0; n * n];
            for row in 0..n {
                for column in 0..n {
                    a[row * n + column] = random.next_f64();
                }
                a[row * n + row] += n as f64;
            }
            let x_true: Vec<f64> = (0..n).map(|_| random.next_f64()).collect();
            let mut b = vec![0.0; n];
            for row in 0..n {
                b[row] = (0..n)
                    .map(|column| a[row * n + column] * x_true[column])
                    .sum();
            }
            let lu = LuFactorization::decompose(&a, n).unwrap();
            let x = lu.solve(&b).unwrap();
            // Backward stability of partial pivoting puts the residual
            // at machine-epsilon level; 1e-12 leaves four orders of
            // magnitude of headroom for n = 50 growth.
            let residual = relative_residual(&a, &x, &b, n);
            assert!(
                residual < 1e-12,
                "n = {n}: relative residual {residual:e} exceeds the bound"
            );
            let error = x.iter().zip(&x_true).fold(0.0f64, |max, (solved, truth)| {
                max.max((solved - truth).abs())
            });
            assert!(
                error < 1e-6,
                "n = {n}: solution error {error:e} against the constructed x"
            );
        }
    }

    #[test]
    fn singular_matrices_are_refused() {
        // For x = y and y = x, I-A = [[1, -1], [-1, 1]], which is singular.
        let singular = [1.0, -1.0, -1.0, 1.0];
        let LinearSolveError::Singular(error) =
            LuFactorization::decompose(&singular, 2).unwrap_err()
        else {
            panic!("expected singularity")
        };
        assert_eq!(error.step, 1);
        // The self-reference x = x: the 1x1 zero matrix.
        let LinearSolveError::Singular(error) = LuFactorization::decompose(&[0.0], 1).unwrap_err()
        else {
            panic!("expected singularity")
        };
        assert_eq!(error.step, 0);
        assert_eq!(error.magnitude, 0.0);
        // The zero matrix at any size.
        let zero = vec![0.0; 9];
        assert!(LuFactorization::decompose(&zero, 3).is_err());
    }

    #[test]
    fn a_nan_entry_is_refused() {
        let poisoned = [1.0, f64::NAN, 2.0, 1.0];
        assert!(LuFactorization::decompose(&poisoned, 2).is_err());
    }

    #[test]
    fn scale_invariance_of_the_relative_test() {
        // The same nearly-singular system, scaled by 1e-30: the
        // relative test must refuse both equally (an absolute floor
        // would pass the small one).
        let matrix = [1.0, 1.0, 1.0, 1.0 + 1e-14];
        assert!(LuFactorization::decompose(&matrix, 2).is_err());
        let tiny: Vec<f64> = matrix.iter().map(|value| value * 1e-30).collect();
        assert!(LuFactorization::decompose(&tiny, 2).is_err());
    }
    #[test]
    fn rejects_nonfinite_with_zero_multiplier() {
        assert!(LuFactorization::decompose(&[1.0, f64::NAN, 0.0, 1.0], 2).is_err());
    }
    #[test]
    fn rejects_invalid_tolerance() {
        for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(LuFactorization::decompose_with_tolerance(&[1.0], 1, tolerance).is_err());
        }
    }
    #[test]
    fn malformed_dimensions_do_not_panic() {
        let result = std::panic::catch_unwind(|| LuFactorization::decompose(&[1.0], 2));
        assert!(result.is_ok());
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn malformed_rhs_does_not_panic() {
        let lu = LuFactorization::decompose(&[1.0], 1).unwrap();
        let result = std::panic::catch_unwind(|| lu.solve_into(&mut []));
        assert!(result.is_ok());
        assert!(result.unwrap().is_err());
    }
    #[test]
    fn invalid_inputs_have_typed_diagnostics() {
        assert!(matches!(
            LuFactorization::decompose(&[], usize::MAX),
            Err(LinearSolveError::DimensionOverflow { .. })
        ));
        assert!(matches!(
            LuFactorization::decompose(&[1.0], 2),
            Err(LinearSolveError::DimensionMismatch {
                input: "matrix",
                expected: 4,
                actual: 1,
            })
        ));
        for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                LuFactorization::decompose_with_tolerance(&[1.0], 1, tolerance),
                Err(LinearSolveError::InvalidTolerance { .. })
            ));
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for index in 0..4 {
                let mut matrix = [1.0, 0.0, 0.0, 1.0];
                matrix[index] = value;
                assert!(matches!(
                    LuFactorization::decompose(&matrix, 2),
                    Err(LinearSolveError::NonFinite { input: "matrix", index: bad }) if bad == index
                ));
            }
        }
    }

    #[test]
    fn rhs_errors_are_typed_and_leave_input_untouched() {
        let lu = LuFactorization::decompose(&[0.0, 1.0, 1.0, 0.0], 2).unwrap();
        assert!(matches!(
            lu.solve(&[1.0]),
            Err(LinearSolveError::DimensionMismatch {
                input: "right-hand side",
                ..
            })
        ));
        let mut too_long = [1.0, 2.0, 3.0];
        assert!(lu.solve_into(&mut too_long).is_err());
        assert_eq!(too_long, [1.0, 2.0, 3.0]);
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut rhs = [3.0, value];
            let before = rhs.map(f64::to_bits);
            assert!(matches!(
                lu.solve_into(&mut rhs),
                Err(LinearSolveError::NonFinite {
                    input: "right-hand side",
                    index: 1
                })
            ));
            assert_eq!(rhs.map(f64::to_bits), before);
        }
    }

    #[test]
    fn repeated_solves_reuse_the_factorization_and_output_buffer() {
        let lu = LuFactorization::decompose(&[0.0, 1.0, 2.0, 3.0], 2).unwrap();
        let factors = lu.lu.clone();
        let factors_pointer = lu.lu.as_ptr();
        let pivots_pointer = lu.pivots.as_ptr();
        let mut rhs = [0.0; 2];
        let output_pointer = rhs.as_ptr();
        for i in 1..100 {
            rhs.copy_from_slice(&[i as f64, 5.0 * i as f64]);
            lu.solve_into(&mut rhs).unwrap();
            assert_eq!(rhs, [i as f64, i as f64]);
            assert_eq!(rhs.as_ptr(), output_pointer);
            assert_eq!(lu.lu.as_ptr(), factors_pointer);
            assert_eq!(lu.pivots.as_ptr(), pivots_pointer);
            assert_eq!(lu.lu, factors);
        }
    }

    #[test]
    fn finite_inputs_that_overflow_are_refused() {
        assert!(matches!(
            LuFactorization::decompose(&[f64::MAX, f64::MAX, -f64::MAX, f64::MAX], 2),
            Err(LinearSolveError::NonFinite {
                input: "factorization",
                ..
            })
        ));
        let lu = LuFactorization::decompose(&[0.5], 1).unwrap();
        assert!(matches!(
            lu.solve_into(&mut [f64::MAX]),
            Err(LinearSolveError::NonFinite {
                input: "solution",
                index: 0
            })
        ));
    }

    #[test]
    fn relative_pivot_floor_is_inclusive_and_scale_invariant() {
        let matrix = [1.0, 0.0, 0.0, 0.5];
        for scale in [1e-300, 1.0, 1e300] {
            let scaled = matrix.map(|value| value * scale);
            assert!(matches!(
                LuFactorization::decompose_with_tolerance(&scaled, 2, 0.5),
                Err(LinearSolveError::Singular(SingularMatrix { step: 1, .. }))
            ));
            assert!(LuFactorization::decompose_with_tolerance(&scaled, 2, 0.49).is_ok());
        }
    }

    #[test]
    fn empty_system_has_an_empty_solution() {
        let lu = LuFactorization::decompose(&[], 0).unwrap();
        assert_eq!(lu.size(), 0);
        assert_eq!(lu.solve(&[]).unwrap(), Vec::<f64>::new());
        lu.solve_into(&mut []).unwrap();
    }
}
