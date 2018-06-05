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

#ifndef RTNEURON_OBJECTCACHE_H
#define RTNEURON_OBJECTCACHE_H

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <set>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/* Thread-safe cache with automatic object creation when needed. */
template <typename Object, typename Key,
          template <typename T> typename Pointer = std::shared_ptr>
class ObjectCache
{
    using ObjectPtr = Pointer<Object>;

public:
    /** Get or create an object from the cache.
        The object will be created invoking the constructor that matches the
        template arguments provided.
        @param key Lookup name.
        @param args Constructor arguments in case the object needs to be
               created.
        @return A newly created or cached object
    */
    template <typename... Args>
    ObjectPtr getOrCreate(const Key& key, Args&&... args)
    {
        ObjectPtr& object = _getOrLockForCreation(key);
        if (object)
            return object;

        Unlock unlock(this, key);
        /* Assignment needed, do not merge with return */
        object = std::make_shared<Object>(std::forward<Args>(args)...);
        return object;
    }

    /** Get or create an object from the cache.
        The object will be created invoking the provided factory function.
        Exception are propagated without doing any visible modification of the
        cache.
        @param key Lookup name.
        @param args Factory arguments in case the object needs to be created.
        @return A newly created or cached object
    */
    template <typename... Args, typename... FArgs>
    ObjectPtr getOrCreate(const Key& key, ObjectPtr factory(FArgs...),
                          Args&&... args)
    {
        ObjectPtr& object = _getOrLockForCreation(key);
        if (object)
            return object;

        Unlock unlock(this, key);
        /* Assignment needed, do not merge with return */
        object = factory(std::forward<Args>(args)...);
        return object;
    }

    void clear()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _objects.clear();
    }

private:
    using ObjectMap = std::map<Key, ObjectPtr>;

    struct Unlock
    {
        Unlock(ObjectCache* o, const Key& key)
            : _o(o)
            , _key(key)
        {
        }
        ~Unlock()
        {
            std::unique_lock<std::mutex> lock(_o->_mutex);
            _o->_underConstruction.erase(_key);
            _o->_createCondition.notify_all();
        }
        ObjectCache* _o;
        const Key& _key;
    };

    std::mutex _mutex;

    ObjectMap _objects;
    std::set<Key> _underConstruction;
    std::condition_variable _createCondition;

    ObjectPtr& _getOrLockForCreation(const Key& key)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _createCondition.wait(lock, [&]() {
            return _underConstruction.find(key) == _underConstruction.end();
        });

        ObjectPtr& object = _objects[key];
        if (!object)
            _underConstruction.insert(key);
        return object;
    }
};

/* Specicialization for single objects, no key */
template <typename Object, template <typename T> typename Pointer>
class ObjectCache<Object, void, Pointer>
{
    using ObjectPtr = Pointer<Object>;

public:
    ObjectCache()
        : _flag(new std::once_flag)
    {
    }

    /** Get or create an object from the cache.
        The object will be created invoking the constructor that matches the
        template arguments provided.
        @param key Lookup name.
        @param args Constructor arguments in case the object needs to be
               created.
        @return A newly created or cached object
    */
    template <typename... Args>
    ObjectPtr getOrCreate(Args&&... args)
    {
        auto flag = _flag;
        std::call_once(*flag,
                       [this](Args&&... argv) {
                           _object = std::make_shared<Object>(argv...);
                       },
                       std::forward<Args>(args)...);
        return _object;
    }

    /** Get or create an object from the cache.
        The object will be created invoking the provided factory function.
        Exception are propagated without doing any visible modification of the
        cache.
        @param key Lookup name.
        @param args Factory arguments in case the object needs to be created.
        @return A newly created or cached object
    */
    template <typename... Args, typename... FArgs>
    ObjectPtr getOrCreate(ObjectPtr factory(FArgs...), Args&&... args)
    {
        auto flag = _flag;
        std::call_once(*flag,
                       [this](ObjectPtr f(FArgs...), Args&&... argv) {
                           _object = f(std::forward<Args>(argv)...);
                       },
                       factory, std::forward<Args>(args)...);
        return _object;
    }

    void clear() { _flag.reset(new std::once_flag); }
private:
    ObjectPtr _object;
    std::shared_ptr<std::once_flag> _flag;
};
}
}
}
#endif
