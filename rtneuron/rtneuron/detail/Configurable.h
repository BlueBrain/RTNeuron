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

#ifndef RTNEURON_API_DETAIL_CONFIGURABLE_H
#define RTNEURON_API_DETAIL_CONFIGURABLE_H

#include "../AttributeMap.h"

namespace bbp
{
namespace rtneuron
{
namespace detail
{
class Configurable
{
public:
    Configurable(const AttributeMap& attributes = AttributeMap())
        : _attributes(attributes)
        , _blockSignals(false)
    {
        /* Connecting to the attribute changed signal */
        _attributes.attributeMapChanged.connect(
            boost::bind(&Configurable::onAttributeChanged, this, _1, _2));
        _attributes.attributeMapChanging.connect(
            boost::bind(&Configurable::onAttributeChanging, this, _1, _2, _3));
    }

    virtual ~Configurable()
    {
        /* No need to disconnect the attribute map signals.
           This object owns the attribute map */
    }

    AttributeMap& getAttributes() { return _attributes; }
    const AttributeMap& getAttributes() const { return _attributes; }
protected:
    /*--- Protected member functions ---*/

    void blockAttributeMapSignals() { _blockSignals = true; }
    void unblockAttributeMapSignals() { _blockSignals = false; }
    /* Attribute change slots */

    void onAttributeChanged(const AttributeMap& attributes,
                            const std::string& name)
    {
        if (!_blockSignals)
            onAttributeChangedImpl(attributes, name);
    }

    void onAttributeChanging(const AttributeMap& attributes,
                             const std::string& name,
                             const AttributeMap::AttributeProxy& parameters)
    {
        if (!_blockSignals)
            onAttributeChangingImpl(attributes, name, parameters);
    }

    virtual void onAttributeChangedImpl(const AttributeMap& attributes,
                                        const std::string& name) = 0;

    virtual void onAttributeChangingImpl(const AttributeMap&,
                                         const std::string& /* name */,
                                         const AttributeMap::AttributeProxy&){};

    /*--- Private member attributes ---*/
private:
    AttributeMap _attributes;
    bool _blockSignals;
};
}
}
}
#endif
