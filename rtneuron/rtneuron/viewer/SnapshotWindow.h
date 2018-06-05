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

#ifndef RTNEURON_SNAPSHOTWINDOW_H
#define RTNEURON_SNAPSHOTWINDOW_H

#include "viewer/osgEq/AuxiliaryWindow.h"
#include "viewer/osgEq/View.h"

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
class View;
}

/**
   The SnapshotWindow is in charge of creating an auxiliary window
   for generating a snapshot of the existing main window in a different
   resolution. This window can be on or off-screen depending on the user's
   requirements.
*/
class SnapshotWindow : public osgEq::AuxiliaryWindow
{
public:
    /** @copydoc osgEq::AuxiliaryWindow::AuxiliaryWindow */
    SnapshotWindow(osgEq::View* view, const eq::fabric::IAttribute& attribute);

    /**
       Sets the filename from the snapshot

       Throws if filename is empty.

       @param filename Full path of the snapshot file
    */
    void setFilename(const std::string& filename);

    /**
       Returns the full path of the snapshot file

       @return full path of the snapshot file
    */
    const std::string& getFilename() const;

protected:
    std::string _filename; /* File name for the snapshot */
    osgEq::View* _auxView;

    /**
       Copies the view parameters from the main view.

       @param config Config object needed to retrieve the additional view.
     */
    void configUpdated(eq::Config* config) final;

    /**
       Renders and takes a snapshot of the additional view.
     */
    void renderImpl(const bool waitForCompletion) final;
};
}
}
#endif
