/* linalg/gsl_linalg.h
 * 
 * Copyright (C) 1996, 1997, 1998, 1999, 2000 Gerard Jungman, Brian Gough
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

#ifndef __GSL_LINALG_H__
#define __GSL_LINALG_H__

#include "gsl_mode.h"
#include "gsl_permutation.h"
#include "gsl_vector.h"
#include "gsl_matrix.h"

typedef enum
  {
    GSL_LINALG_MOD_NONE = 0,
    GSL_LINALG_MOD_TRANSPOSE = 1,
    GSL_LINALG_MOD_CONJUGATE = 2
  }
gsl_linalg_matrix_mod_t;


int gsl_linalg_solve_tridiag (const gsl_vector * diag,
                              const gsl_vector * abovediag,
                              const gsl_vector * belowdiag,
                              const gsl_vector * b,
                              gsl_vector * x);

int gsl_linalg_solve_symm_tridiag(const gsl_vector * diag,
                                  const gsl_vector * offdiag,
                                  const gsl_vector * rhs,
                                  gsl_vector * solution);

int gsl_linalg_solve_symm_cyc_tridiag(const gsl_vector * diag,
                                      const gsl_vector * offdiag,
                                      const gsl_vector * rhs,
                                      gsl_vector * solution);

#endif /* __GSL_LINALG_H__ */
