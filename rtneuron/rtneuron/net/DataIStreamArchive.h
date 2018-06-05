/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Daniel Nachbaur <danielnachbaur@googlemail.com>
 *                          Stefan Eilemann <eile@eyescale.ch>
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

// Based on portable_iarchive.hpp
// https://github.com/boost-vault/serialization/eos_portable_archive.zip
// Copyright Christian Pfligersdorffer, 2007. All rights reserved.
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <co/types.h>

#include <boost/version.hpp>
#pragma warning(push)
#pragma warning(disable : 4800)
#include <boost/archive/basic_binary_iarchive.hpp>
#pragma warning(pop)
#include <boost/archive/detail/register_archive.hpp>
#if BOOST_VERSION < 105600
#include <boost/archive/shared_ptr_helper.hpp>
#endif
#include <boost/serialization/is_bitwise_serializable.hpp>

#include <boost/spirit/home/support/detail/endian.hpp>
#include <boost/spirit/home/support/detail/math/fpclassify.hpp>

#include <boost/type_traits/is_floating_point.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_unsigned.hpp>

#include <boost/utility/enable_if.hpp>

namespace bbp
{
namespace rtneuron
{
namespace net
{
/** A boost.serialization input archive reading from a co::DataIStream. */
class DataIStreamArchive
    : public boost::archive::basic_binary_iarchive<DataIStreamArchive>
#if BOOST_VERSION < 105600
      ,
      public boost::archive::detail::shared_ptr_helper
#endif
{
    typedef boost::archive::basic_binary_iarchive<DataIStreamArchive> Super;

#if BOOST_VERSION >= 106000
    template <typename T>
    using array_serializer = boost::serialization::array_wrapper<T>;
#else
    template <typename T>
    using array_serializer = boost::serialization::array<T>;
#endif

public:
    /** Construct a new deserialization archive. @version 1.0 */
    explicit DataIStreamArchive(co::DataIStream& stream);

    /** @internal archives are expected to support this function */
    void load_binary(void* data, std::size_t size);

    /** @internal use optimized load for arrays. */
    template <typename T>
    void load_array(array_serializer<T>& a, unsigned int);

    /** @internal enable serialization optimization for arrays. */
    struct use_array_optimization
    {
        template <class T>
        struct apply : public boost::serialization::is_bitwise_serializable<T>
        {
        };
    };

private:
    friend class boost::archive::load_access;

    /**
     * Load boolean.
     *
     * Special case loading bool type, preserving compatibility to integer
     * types - this is somewhat redundant but simply treating bool as integer
     * type generates lots of warnings.
     */
    void load(bool& b);

    /** Load string types. */
    template <class C, class T, class A>
    void load(std::basic_string<C, T, A>& s);

    /**
     * Load integer types.
     *
     * First we load the size information ie. the number of bytes that
     * hold the actual data. Then we retrieve the data and transform it
     * to the original value by using load_little_endian.
     */
    template <typename T>
    typename boost::enable_if<boost::is_integral<T>>::type load(T& t);

    /**
     * Load floating point types.
     *
     * We simply rely on fp_traits to set the bit pattern from the (unsigned)
     * integral type that was stored in the stream. Francois Mauger provided
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
    typename boost::enable_if<boost::is_floating_point<T>>::type load(T& t);

    void load(boost::archive::library_version_type& version);
    void load(boost::archive::class_id_type& class_id);
    void load(boost::serialization::item_version_type& version);
    void load(boost::serialization::collection_size_type& version);
    void load(boost::archive::object_id_type& object_id);
    void load(boost::archive::version_type& version);

    signed char _loadSignedChar();

    co::DataIStream& _stream;
};
}
}
}

#include "DataIStreamArchive.ipp" // template implementation

// contains load_override impl for class_name_type
#include <boost/archive/impl/basic_binary_iarchive.ipp>

BOOST_SERIALIZATION_REGISTER_ARCHIVE(bbp::rtneuron::net::DataIStreamArchive)
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(
    bbp::rtneuron::net::DataIStreamArchive)
