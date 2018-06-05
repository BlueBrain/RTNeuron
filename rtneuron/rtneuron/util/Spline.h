/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Juan Hernando <juan.hernando@epfl.ch>
 *
 * This file is part of RTNeuron <https://github.com/BlueBrain/RTNeuron>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef SPLINE_H
#define SPLINE_H

#include <boost/shared_ptr.hpp>
#include <iostream>

namespace bbp
{
namespace rtneuron
{
namespace core
{
//! Wrapper class around the gsl spline functions.
class Spline
{
    /* Contructors */
public:
    Spline();

    ~Spline();

    /* Member functions */
public:
    /**
       Assignment operator.
     */
    Spline& operator=(const Spline& s);

    /**
       Sets the data point from wich the spline is calculated.
       @param x The x coordinates of the points to interpolate.
       @param y The y coordinates of the points to interpolate.
       @param length The length of both the x and y arrays.
     */
    void setData(const double* x, const double* y, int length);

    /**
       Returns the evaluation of the spline in x.
     */
    double eval(double x);

    /**
       Returns the evaluation of the derivative of the spline in x.
     */
    double evalDeriv(double x);

private:
    /**
     */
    void freeSpline();

    /* Member attributes */
private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};
}
}
}
#endif
