/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#include <eq/gl.h> /* To include gl.h before glew.h */

#include "SnapshotWindow.h"

#include "AttributeMap.h"
#include "Camera.h"
#include "View.h"

#include "viewer/osgEq/Application.h"
#include "viewer/osgEq/Client.h"

#include <eq/admin/view.h>
#include <eq/config.h>

namespace bbp
{
namespace rtneuron
{
SnapshotWindow::SnapshotWindow(osgEq::View* view,
                               const eq::fabric::IAttribute& attribute)
    : AuxiliaryWindow(view, attribute)
    , _auxView(0)
{
}

void SnapshotWindow::setFilename(const std::string& filename)
{
    if (filename.empty())
        throw std::runtime_error(
            "A file name for the snapshot "
            "must be specified.");
    _filename = filename;
}

const std::string& SnapshotWindow::getFilename() const
{
    return _filename;
}

void SnapshotWindow::configUpdated(eq::Config* config)
{
    /* Retrieve snapshot view from Equalizer configuration */
    _auxView =
        static_cast<osgEq::View*>(config->find<eq::View>(_admin.view->getID()));
    LBASSERT(_auxView);

    eq::ClientPtr eqClient = config->getClient();
    osgEq::Client* client = static_cast<osgEq::Client*>(eqClient.get());

    _auxView->copyAttributes(*_mainView);

    /* Trigerring one frame to ensure than the channel is ready when
       snapshot is called. Currently, this has the side effect of rendering
       a frame also in the original view. */
    client->frame();
    client->waitFrame();
}

void SnapshotWindow::renderImpl(const bool waitForCompletion)
{
    LBASSERT(_auxView);
    _auxView->snapshot(_filename, waitForCompletion);
}
}
}
