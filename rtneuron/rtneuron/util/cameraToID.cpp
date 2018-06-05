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

#include "cameraToID.h"

#include <osg/Camera>
#include <osg/Observer>
#include <osg/RenderInfo>
#include <osg/ValueObject>

#include <eq/fabric/eye.h>

#include <lunchbox/bitOperation.h>
#include <lunchbox/lfVector.h>

#include <mutex>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
class IDSet : public osg::Observer
{
public:
    /*--- Public member functions ---*/

    uint32_t getID(const osg::Referenced* object)
    {
        /* Searching the object and the next free slot at the same time */
        uint32_t index = 0;
        for (References::iterator i = _references.begin();
             i != _references.end(); ++i, ++index)
        {
            if (*i == object)
                return index;
        }

        /* ID not found using the fully synchronized method */
        return _getIDSynchronized(object);
    }

    virtual void objectDeleted(void* object)
    {
        /* This method is only synchronized with the assignment of IDs to
           new objects. It's possible for getID to return and ID just before
           the associated object is dereference from here. Despite
           some execution orders may seem inconsistent with the result,
           trying to get an ID of a object being destroyed is a client
           programming error because addObserver could be called in some
           valid execution orders (regardless of the thread-safety of IDSet). */
        std::lock_guard<std::mutex> lock(_mutex);

        References::iterator i = _references.begin();
        while (i != _references.end() && *i != object)
            ++i;

        assert(i != _references.end());
        *i = 0;
    }

private:
    /*--- Private member attributes ---*/

    std::mutex _mutex;

    typedef lunchbox::LFVector<const osg::Referenced*> References;
    References _references;

    /*--- Private member functions ---*/

    uint32_t _getIDSynchronized(const osg::Referenced* object)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        uint32_t nextFree = _references.size();
        /* Searching the object and the next free slot at the same time.
           The reference to object might have been added while the calling
           getID was searching for it, thus we need to search again before
           assuming it doesn't exist. */
        for (uint32_t i = 0; i != _references.size(); ++i)
        {
            const osg::Referenced* o = _references[i];
            if (o == 0)
                nextFree = i;
            if (o == object)
                return i;
        }

        /* New object */
        if (nextFree == _references.size())
        {
            /* No free ID available resizing the vector.
               lunchbox::lfVector guarantees that this is safe with
               concurrent calls to getID. */
            _references.expand(nextFree + 1);
        }
        object->addObserver(this);
        _references[nextFree] = object;
        return nextFree;
    }
};

std::mutex s_initMutex;
IDSet& getIDSet()
{
    std::lock_guard<std::mutex> lock(s_initMutex);
    static IDSet idset;
    return idset;
}
}

uint32_t getUniqueCameraID(const osg::Camera* camera)
{
    /* The indirection is used to guarantee that the init mutex is only
       acquired once, during the initialization of the static variable.
       The initialization of the reference itself is not thread-safe but
       there's no danger in two threads doing it at the same time. */
    static IDSet& idSet = getIDSet();
    return idSet.getID(camera);
}

uint32_t getCameraAndEyeID(const osg::Camera* camera)
{
    eq::fabric::Eye eye = eq::fabric::EYE_CYCLOP;
    unsigned int eyei = (unsigned int)eye;
    if (camera->getUserValue("eye", eyei))
        eye = (eq::fabric::Eye)eyei;
    uint32_t index = getUniqueCameraID(camera) * uint32_t(eq::fabric::NUM_EYES);
    index += lunchbox::getIndexOfLastBit(eye);
    return index;
}

uint32_t getCameraAndEyeID(osg::RenderInfo& info)
{
    const osg::Camera* camera = info.getCurrentCamera();
    return getCameraAndEyeID(camera);
}
}
}
}
