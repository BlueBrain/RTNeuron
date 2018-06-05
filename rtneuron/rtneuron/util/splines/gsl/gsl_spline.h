/* interpolation/gsl_spline.h
 * 
 * Copyright (C) 2001 Brian Gough
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

#ifndef __GSL_SPLINE_H__
#define __GSL_SPLINE_H__
#include <stdlib.h>
#include "gsl_interp.h"

/* general interpolation object */
typedef struct {
  gsl_interp * interp;
  double  * x;
  double  * y;
  size_t  size;
} gsl_spline;

gsl_spline *
gsl_spline_alloc(const gsl_interp_type * T, size_t size);
     
int
gsl_spline_init(gsl_spline * spline, const double xa[], const double ya[], size_t size);


int
gsl_spline_eval_e(const gsl_spline * spline, double x,
                  gsl_interp_accel * a, double * y);

double
gsl_spline_eval(const gsl_spline * spline, double x, gsl_interp_accel * a);

int
gsl_spline_eval_deriv_e(const gsl_spline * spline,
                        double x,
                        gsl_interp_accel * a,
                        double * y);

double
gsl_spline_eval_deriv(const gsl_spline * spline,
                      double x,
                      gsl_interp_accel * a);

int
gsl_spline_eval_deriv2_e(const gsl_spline * spline,
                         double x,
                         gsl_interp_accel * a,
                         double * y);

double
gsl_spline_eval_deriv2(const gsl_spline * spline,
                       double x,
                       gsl_interp_accel * a);

int
gsl_spline_eval_integ_e(const gsl_spline * spline,
                        double a, double b,
                        gsl_interp_accel * acc,
                        double * y);

double
gsl_spline_eval_integ(const gsl_spline * spline,
                      double a, double b,
                      gsl_interp_accel * acc);

void
gsl_spline_free(gsl_spline * spline);

#endif /* __GSL_INTERP_H__ */
