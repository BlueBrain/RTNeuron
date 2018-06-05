//////////////////////////////////////////////////////////////////////
// RTNeuron
//
// Copyright (c) 2006-2013 Cajal Blue Brain, BBP/EPFL
// All rights reserved. Do not distribute without permission.
//
// Responsible Author: Juan Hernando Vieites (JHV)
// contact: jhernando@fi.upm.es
//////////////////////////////////////////////////////////////////////

#include "AutoExpandLFVector.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{

template<typename T>
T &AutoExpandLFVector<T>::operator[](size_t index)
{
    const T &value =
        static_cast<const AutoExpandLFVector<T>&>(*this).operator[](index);
    return const_cast<T&>(value);
}

template<typename T>
const T &AutoExpandLFVector<T>::operator[](size_t index) const
{
    if (_vector.size() <= index)
        _vector.expand(index + 1);
    return _vector[index];
}

}
}
}
