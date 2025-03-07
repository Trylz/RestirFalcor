#pragma once

#include "Falcor.h"

namespace Restir
{
template<typename InstanceType>
class Singleton
{
public:
    Singleton() = delete;

    static InstanceType* create()
    {
        FMT_ASSERT(m_instance == nullptr, "");
        m_instance = new InstanceType();

        return m_instance;
    }

    static void destroy()
    {
        FMT_ASSERT(m_instance != nullptr, "");
        delete m_instance;
        m_instance = nullptr;
    }

    static InstanceType* instance() { return m_instance; }

private:
    static InstanceType* m_instance;
};

template<typename InstanceType>
InstanceType* Singleton<InstanceType>::m_instance = nullptr;
} // namespace Restir
