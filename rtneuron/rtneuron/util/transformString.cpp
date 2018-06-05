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

#include <osg/Matrix>
#include <osg/Vec3>

#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>

namespace bbp
{
namespace rtneuron
{
namespace core
{
osg::Matrix parseTranslateRotateScaleString(std::string string)
{
    if (string == "")
        return osg::Matrix();

    char separator = ':';
    std::list<std::string> transforms;
    std::list<std::string>::iterator iter_trans;
    size_t pos = string.find_first_of(separator);
    while (pos != std::string::npos)
    {
        /* Store transform in list. */
        transforms.push_back(string.substr(0, pos));
        /* Did we hit end of string? */
        if (pos + 1 >= string.size())
            break;
        /* Remove string we have stored */
        string = string.substr(pos + 1);
        /* find end of next transform */
        pos = string.find_first_of(separator);
    }
    /* Last transform is in working_string */
    transforms.push_back(string);

    std::string trans_type;
    std::stringstream data;

    // Initial matrix transform
    osg::Matrix m = osg::Matrix::identity();

    for (iter_trans = transforms.begin(); iter_trans != transforms.end();
         ++iter_trans)
    {
        if (iter_trans->size() < 2)
        {
            std::stringstream err;
            err << "RTNeuron: Error parsing transformation: " << *iter_trans
                << std::endl;
            throw std::invalid_argument(err.str().c_str());
        }
        trans_type = iter_trans->substr(0, 2);
        data.str(iter_trans->substr(2));
        data.clear();
        char _;
        if (trans_type == "r@")
        {
            /* Rotation */
            osg::Quat rotation;
            osg::Vec3 axis;
            double angle;
            data >> axis[0] >> _ >> axis[1] >> _ >> axis[2] >> _ >> angle;
            rotation.makeRotate(angle / 180 * M_PI, axis);
            if (data.fail())
            {
                std::stringstream err;
                err << "RTNeuron: Error parsing rotation argument: "
                    << data.str() << std::endl;
                throw std::invalid_argument(err.str().c_str());
            }
            else
                m.postMultRotate(rotation);
        }
        else if (trans_type == "t@")
        {
            /* Translation */
            osg::Vec3 translation(0, 0, 0);
            data >> translation[0] >> _ >> translation[1] >> _ >>
                translation[2];
            if (data.fail())
            {
                std::stringstream err;
                err << "RTNeuron: Error parsing translation argument: "
                    << data.str() << std::endl;
                throw std::invalid_argument(err.str().c_str());
            }
            else
                m.postMultTranslate(translation);
        }
        else if (trans_type == "s@")
        {
            /* Scaling */
            osg::Vec3 scaling;
            data >> scaling[0] >> _ >> scaling[1] >> _ >> scaling[2];
            if (data.fail())
            {
                std::stringstream err;
                err << "RTNeuron: Error parsing scaling argument: "
                    << data.str() << std::endl;
                throw std::invalid_argument(err.str().c_str());
            }
            else
                m.postMultScale(scaling);
        }
        else
        {
            std::stringstream err;
            err << "RTNeuron: Error parsing transformation: " << *iter_trans
                << std::endl;
            throw std::invalid_argument(err.str().c_str());
        }
    }

    return m;
}
}
}
}
