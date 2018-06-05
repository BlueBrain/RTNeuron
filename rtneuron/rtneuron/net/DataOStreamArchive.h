/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Daniel Nachbaur <danielnachbaur@googlemail.com>
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

// Based on portable_oarchive.hpp
// https://github.com/boost-vault/serialization/eos_portable_archive.zip
// Copyright Christian Pfligersdorffer, 2007. All rights reserved.
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "DataStreamArchiveException.h"
#include <co/dataOStream.h>

#include <boost/version.hpp>

#include <boost/archive/basic_binary_oarchive.hpp>
#include <boost/archive/detail/register_archive.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
#if BOOST_VERSION >= 104400
#include <boost/serialization/item_version_type.hpp>
#endif

#include <boost/spirit/home/support/detail/endian.hpp>
#include <boost/spirit/home/support/detail/math/fpclassify.hpp>

#include <boost/type_traits/is_floating_point.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_signed.hpp>

namespace bbp
{
namespace rtneuron
{
namespace net
{
/** A boost.serialization output archive writing to a co::DataOStream. */
class DataOStreamArchive
    : public boost::archive::basic_binary_oarchive<DataOStreamArchive>
{
    typedef boost::archive::basic_binary_oarchive<DataOStreamArchive> Super;

#if BOOST_VERSION >= 106000
    template <typename T>
    using array_serializer = boost::serialization::array_wrapper<T>;
#else
    template <typename T>
    using array_serializer = boost::serialization::array<T>;
#endif

public:
    /** Construct a new serialization archive. @version 1.0 */
    CO_API explicit DataOStreamArchive(co::DataOStream& stream);

    /** @internal archives are expected to support this function. */
    CO_API void save_binary(const void* data, std::size_t size);

    /** @internal use optimized save for arrays. */
    template <typename T>
    void save_array(const array_serializer<T>& a, unsigned int);

    /** @internal enable serialization optimization for arrays. */
    struct use_array_optimization
    {
        template <class T>
        struct apply : public boost::serialization::is_bitwise_serializable<T>
        {
        };
    };

private:
    friend class boost::archive::save_access;

    /**
     * Save boolean.
     *
     * Saving bool directly, not by const reference because of tracking_type's
     * operator (bool).
     */
    CO_API void save(bool b);

    /** Save string types. */
    template <class C, class T, class A>
    void save(const std::basic_string<C, T, A>& s);

    /**
     * Save integer types.
     *
     * First we save the size information ie. the number of bytes that hold the
     * actual data. We subsequently transform the data using store_little_endian
     * and store non-zero bytes to the stream.
     */
    template <typename T>
    typename boost::enable_if<boost::is_integral<T>>::type save(const T& t);

    /**
     * Save floating point types.
     *
     * We simply rely on fp_traits to extract the bit pattern into an (unsigned)
     * integral type and store that into the stream. Francois Mauger provided
     * standardized behaviour for special values like inf and NaN, that need to
     * be serialized in his application.
     *
     * \note by Johan Rade (author of the floating point utilities library):
     * Be warned that the math::detail::fp_traits<T>::type::get_bits() function
     * is *not* guaranteed to give you all bits of the floating point number. It
     * will give you all bits if and only if there is an integer type that has
     * the same size as the floating point you are copying from. It will not
     * give you all bits for double if there is no uint64_t. It will not give
     * you all bits for long double if sizeof(long double) > 8 or there is no
     * uint64_t.
     *
     * The member fp_traits<T>::type::coverage will tell you whether all bits
     * are copied. This is a typedef for either math::detail::all_bits or
     * math::detail::not_all_bits.
     *
     * If the function does not copy all bits, then it will copy the most
     * significant bits. So if you serialize and deserialize the way you
     * describe, and fp_traits<T>::type::coverage is math::detail::not_all_bits,
     * then your floating point numbers will be truncated. This will introduce
     * small rounding off errors.
     */
    template <typename T>
    typename boost::enable_if<boost::is_floating_point<T>>::type save(
        const T& t);

#if BOOST_VERSION >= 104400
    // in boost 1.44 version_type was splitted into library_version_type and
    // item_version_type, plus a whole bunch of additional strong typedefs
    CO_API void save(const boost::archive::library_version_type& version);
    CO_API void save(const boost::archive::class_id_type& class_id);
    CO_API void save(const boost::serialization::item_version_type& class_id);
    CO_API
    void save(const boost::serialization::collection_size_type& class_id);
    CO_API void save(const boost::archive::object_id_type& object_id);
    CO_API void save(const boost::archive::version_type& version);
#endif

    CO_API void _saveSignedChar(const signed char& c);

    co::DataOStream& _stream;
};

#include "DataOStreamArchive.ipp" // template implementation
}
}
}

BOOST_SERIALIZATION_REGISTER_ARCHIVE(bbp::rtneuron::net::DataOStreamArchive)
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(
    bbp::rtneuron::net::DataOStreamArchive)
