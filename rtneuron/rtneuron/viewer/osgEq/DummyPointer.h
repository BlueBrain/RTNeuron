/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Jafet Villafranca <jafet.villafrancadiaz@epfl.ch>
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

#ifndef RTNEURON_OSGEQ_DUMMYPOINTER_H
#define RTNEURON_OSGEQ_DUMMYPOINTER_H

#include "ui/Pointer.h"

namespace bbp
{
namespace rtneuron
{
/**
   Derived class used to deserialize the 3D pointer data and update the
   geometry in rendering clients.
 */
class DummyPointer : public Pointer
{
public:
    /*--- Public constructors/destructors ---*/

    DummyPointer(){};

    ~DummyPointer(){};

    /*--- Public member functions ---*/

    void update() {}
};
}
}

#endif /* RTNEURON_OSQEQ_DUMMYPOINTER_H */
