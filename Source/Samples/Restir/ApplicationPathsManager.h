#pragma once

#include "Singleton.h"

namespace Restir
{
class ApplicationPathsManager
{
public:
    ApplicationPathsManager(){};

    void setExePath(const std::string& path) { mExePath = path; }
    void setScenePath(const std::string& path) { mScenePath = path; }
    void setSharedDataPath(const std::string& path) { mSharedDataPath = path; }

    inline const std::string& getExePath() const { return mExePath; }
    inline const std::string& getScenePath() const { return mScenePath; }
    inline const std::string& getSharedDataPath() const { return mSharedDataPath; }

private:
    std::string mExePath;
    std::string mScenePath;
    std::string mSharedDataPath;
};

using ApplicationPathsManagerSingleton = Singleton<ApplicationPathsManager>;

} // namespace Restir

