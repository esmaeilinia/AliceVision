// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/sfm/SfMData.hpp>

#include <string>

namespace aliceVision {
namespace sfm {

/**
 * @brief Basic Reconstruction Engine.
 * Process Function handle the reconstruction.
 */
class ReconstructionEngine
{
public:

  /**
   * @brief ReconstructionEngine Constructor
   * @param[in] sfmData The input SfMData of the scene
   * @param[in] outFolder The folder where outputs will be stored
   */
  ReconstructionEngine(const SfMData& sfmData, const std::string& outFolder)
    : _outputFolder(outFolder)
    , _sfmData(sfmData)
    , _hasFixedIntrinsics(false)
  {}

  virtual ~ReconstructionEngine() {}

  /**
   * @brief Reconstruction process
   * @return true if the scene is reconstructed
   */
  virtual bool process() = 0;

  /**
   * @brief Return true or false the intrinsics are fixed
   * @return true if the intrinsics are fixed
   */
  inline bool hasFixedIntrinsics() const
  {
    return _hasFixedIntrinsics;
  }

  /**
   * @brief Get the scene SfMData
   * @return SfMData
   */
  inline const SfMData& getSfMData() const
  {
    return _sfmData;
  }

  /**
   * @brief Set true or false the intrinsics are fixed
   * @param[in] fixed true if intrinsics are fixed
   */
  inline void setFixedIntrinsics(bool fixed)
  {
    _hasFixedIntrinsics = fixed;
  }

  /**
   * @brief Get the scene SfMData
   * @return SfMData
   */
  inline SfMData& getSfMData()
  {
    return _sfmData;
  }

  /**
   * @brief Colorization of the reconstructed scene
   * @return true if ok
   */
  inline bool colorize()
  {
    return colorizeTracks(_sfmData);
  }

protected:
  /// Output folder where outputs will be stored
  std::string _outputFolder;
  /// Internal SfMData
  SfMData _sfmData;
  /// Has fixed Intrinsics
  bool _hasFixedIntrinsics;
};

} // namespace sfm
} // namespace aliceVision
