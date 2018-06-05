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

#include "Spline.h"

#include "util/splines/gsl/gsl_spline.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
class Spline::Impl
{
public:
    gsl_interp_accel* accel;
    gsl_spline* spline;
    boost::shared_ptr<int> sharedCount;

    Impl()
        : accel(0)
        , spline(0)
    {
    }

    ~Impl() { freeSpline(); }
    void freeSpline()
    {
        if (spline && sharedCount && sharedCount.unique())
        {
            gsl_spline_free(spline);
            gsl_interp_accel_free(accel);
        }
    }
};

Spline::Spline()
    : _impl(new Impl())
{
}

Spline::~Spline()
{
}

Spline& Spline::operator=(const Spline& s)
{
    _impl->freeSpline();
    _impl.reset(new Impl(*s._impl));
    return *this;
}

void Spline::setData(const double* x, const double* y, int length)
{
    _impl->freeSpline();
    _impl->sharedCount.reset(new int());
    _impl->accel = gsl_interp_accel_alloc();
    _impl->spline = gsl_spline_alloc(gsl_interp_cspline, length);
    if (!_impl->accel || !_impl->spline)
    {
        std::cerr << "Out of memory creating spline function" << std::endl;
        abort();
    }
    gsl_spline_init(_impl->spline, x, y, length);
}

double Spline::eval(double x)
{
    return gsl_spline_eval(_impl->spline, x, _impl->accel);
}

double Spline::evalDeriv(double x)
{
    return gsl_spline_eval_deriv(_impl->spline, x, _impl->accel);
}
}
}
}
