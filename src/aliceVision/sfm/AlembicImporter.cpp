// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "AlembicImporter.hpp"

#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreFactory/All.h>
#include <Alembic/AbcCoreOgawa/All.h>

namespace aliceVision {
namespace sfm {

using namespace Alembic::Abc;
using namespace Alembic::AbcGeom;

template<class AbcArrayProperty, typename T>
void getAbcArrayProp(ICompoundProperty& userProps, const std::string& id, index_t sampleFrame, T& outputArray)
{
  typedef typename AbcArrayProperty::sample_ptr_type sample_ptr_type;

  AbcArrayProperty prop(userProps, id);
  sample_ptr_type sample;
  prop.get(sample, ISampleSelector(sampleFrame));
  outputArray.assign(sample->get(), sample->get()+sample->size());
}

/**
 * @brief Retrieve an Abc property.
 *         Maya convert everything into arrays. So here we made a trick
 *         to retrieve the element directly or the first element
 *         if it's an array.
 * @param userProps
 * @param id
 * @param sampleFrame
 * @return value
 */
template<class AbcProperty>
typename AbcProperty::traits_type::value_type getAbcProp(ICompoundProperty& userProps, const Alembic::Abc::PropertyHeader& propHeader, const std::string& id, index_t sampleFrame)
{
  typedef typename AbcProperty::traits_type traits_type;
  typedef typename traits_type::value_type value_type;
  typedef typename Alembic::Abc::ITypedArrayProperty<traits_type> array_type;
  typedef typename array_type::sample_ptr_type array_sample_ptr_type;

  // Maya transforms everything into arrays
  if(propHeader.isArray())
  {
    Alembic::Abc::ITypedArrayProperty<traits_type> prop(userProps, id);
    array_sample_ptr_type sample;
    prop.get(sample, ISampleSelector(sampleFrame));
    return (*sample)[0];
  }
  else
  {
    value_type v;
    AbcProperty prop(userProps, id);
    prop.get(v, ISampleSelector(sampleFrame));
    return v;
  }
}


template<class ABCSCHEMA>
inline ICompoundProperty getAbcUserProperties(ABCSCHEMA& schema)
{
  ICompoundProperty userProps = schema.getUserProperties();
  if(userProps && userProps.getNumProperties() != 0)
    return userProps;

  // Maya always use ArbGeomParams instead of user properties.
  return schema.getArbGeomParams();
}


bool readPointCloud(IObject iObj, M44d mat, sfm::SfMData &sfmdata, sfm::ESfMData flags_part)
{
  using namespace aliceVision::geometry;
  using namespace aliceVision::sfm;

  IPoints points(iObj, kWrapExisting);
  IPointsSchema& ms = points.getSchema();
  P3fArraySamplePtr positions = ms.getValue().getPositions();

  ICompoundProperty userProps = getAbcUserProperties(ms);
  ICompoundProperty arbGeom = ms.getArbGeomParams();

  C3fArraySamplePtr sampleColors;
  if(arbGeom && arbGeom.getPropertyHeader("color"))
  {
    IC3fArrayProperty propColor(arbGeom, "color");
    propColor.get(sampleColors);
    if(sampleColors->size() != positions->size())
    {
      ALICEVISION_LOG_WARNING("[Alembic Importer] Colors will be ignored. Color vector size: " << sampleColors->size() << ", positions vector size: " << positions->size());
      sampleColors.reset();
    }
  }

  UInt32ArraySamplePtr sampleDescs;
  if(userProps && userProps.getPropertyHeader("mvg_describerType"))
  {
    IUInt32ArrayProperty propDesc(userProps, "mvg_describerType");
    propDesc.get(sampleDescs);
    if(sampleDescs->size() != positions->size())
    {
      ALICEVISION_LOG_WARNING("[Alembic Importer] Describer type will be ignored. describerType vector size: " << sampleDescs->size() << ", positions vector size: " << positions->size());
      sampleDescs.reset();
    }
  }

  // Number of points before adding the Alembic data
  const std::size_t nbPointsInit = sfmdata.structure.size();
  for(std::size_t point3d_i = 0;
      point3d_i < positions->size();
      ++point3d_i)
  {
    const P3fArraySamplePtr::element_type::value_type & pos_i = positions->get()[point3d_i];
    Landmark& landmark = sfmdata.structure[nbPointsInit + point3d_i] = Landmark(Vec3(pos_i.x, pos_i.y, pos_i.z), feature::EImageDescriberType::UNKNOWN);

    if(sampleColors)
    {
      const P3fArraySamplePtr::element_type::value_type & color_i = sampleColors->get()[point3d_i];
      landmark.rgb = image::RGBColor(static_cast<unsigned char>(color_i[0] * 255.0f),
                                     static_cast<unsigned char>(color_i[1] * 255.0f),
                                     static_cast<unsigned char>(color_i[2] * 255.0f)
                                     );
    }

    if(sampleDescs)
    {
      const UInt32ArraySamplePtr::element_type::value_type & descType_i = sampleDescs->get()[point3d_i];
      landmark.descType = static_cast<feature::EImageDescriberType>(descType_i);
    }
  }

  if(userProps &&
     userProps.getPropertyHeader("mvg_visibilitySize") &&
     userProps.getPropertyHeader("mvg_visibilityIds") &&
     userProps.getPropertyHeader("mvg_visibilityFeatPos"))
  {
    IUInt32ArrayProperty propVisibilitySize(userProps, "mvg_visibilitySize");
    UInt32ArraySamplePtr sampleVisibilitySize;
    propVisibilitySize.get(sampleVisibilitySize);

    IUInt32ArrayProperty propVisibilityIds(userProps, "mvg_visibilityIds");
    UInt32ArraySamplePtr sampleVisibilityIds;
    propVisibilityIds.get(sampleVisibilityIds);

    IFloatArrayProperty propFeatPos2d(userProps, "mvg_visibilityFeatPos");
    FloatArraySamplePtr sampleFeatPos2d;
    propFeatPos2d.get(sampleFeatPos2d);

    if( positions->size() != sampleVisibilitySize->size() )
    {
      ALICEVISION_LOG_WARNING("ABC Error: number of observations per 3D point should be identical to the number of 2D features.");
      ALICEVISION_LOG_WARNING("Number of observations per 3D point size is " << sampleVisibilitySize->size());
      ALICEVISION_LOG_WARNING("Number of 3D points is " << positions->size());
      return false;
    }
    if( sampleVisibilityIds->size() != sampleFeatPos2d->size() )
    {
      ALICEVISION_LOG_WARNING("ABC Error: visibility Ids and features 2D pos should have the same size.");
      ALICEVISION_LOG_WARNING("Visibility Ids size is " << sampleVisibilityIds->size());
      ALICEVISION_LOG_WARNING("Features 2d Pos size is " << sampleFeatPos2d->size());
      return false;
    }

    std::size_t obsGlobal_i = 0;
    for(std::size_t point3d_i = 0;
        point3d_i < positions->size();
        ++point3d_i)
    {
      Landmark& landmark = sfmdata.structure[nbPointsInit + point3d_i];
      // Number of observation for this 3d point
      const std::size_t visibilitySize = (*sampleVisibilitySize)[point3d_i];

      for(std::size_t obs_i = 0;
          obs_i < visibilitySize*2;
          obs_i+=2, obsGlobal_i+=2)
      {

        const int viewID = (*sampleVisibilityIds)[obsGlobal_i];
        const int featID = (*sampleVisibilityIds)[obsGlobal_i+1];
        Observation& observations = landmark.observations[viewID];
        observations.id_feat = featID;

        const float posX = (*sampleFeatPos2d)[obsGlobal_i];
        const float posY = (*sampleFeatPos2d)[obsGlobal_i+1];
        observations.x[0] = posX;
        observations.x[1] = posY;
      }
    }
  }
  return true;
}

bool readCamera(const ICamera& camera, const M44d& mat, sfm::SfMData& sfmData, sfm::ESfMData flagsPart, const index_t sampleFrame = 0, bool isReconstructed = true)
{
  using namespace aliceVision::geometry;
  using namespace aliceVision::camera;
  using namespace aliceVision::sfm;

  ICameraSchema cs = camera.getSchema();
  CameraSample camSample;
  if(sampleFrame == 0)
    camSample = cs.getValue();
  else
    camSample = cs.getValue(ISampleSelector(sampleFrame));

  // Check if we have an associated image plane
  ICompoundProperty userProps = getAbcUserProperties(cs);
  std::string imagePath;
  std::vector<unsigned int> sensorSize_pix = {0, 0};
  std::string mvg_intrinsicType = EINTRINSIC_enumToString(PINHOLE_CAMERA);
  std::vector<double> mvg_intrinsicParams;
  double initialFocalLengthPix = -1;
  std::vector<std::string> rawMetadata;
  IndexT viewId = sfmData.getViews().size();
  IndexT poseId = sfmData.getViews().size();
  IndexT intrinsicId = sfmData.getIntrinsics().size();
  IndexT rigId = UndefinedIndexT;
  IndexT subPoseId = UndefinedIndexT;
  IndexT resectionId = UndefinedIndexT;
  bool poseLocked = false;
  bool intrinsicLocked = false;

  if(userProps)
  {
    if(flagsPart & sfm::ESfMData::VIEWS || flagsPart & sfm::ESfMData::INTRINSICS)
    {
      if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_imagePath"))
        imagePath = getAbcProp<Alembic::Abc::IStringProperty>(userProps, *propHeader, "mvg_imagePath", sampleFrame);

      if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_viewId"))
      {
        try
        {
          viewId = getAbcProp<Alembic::Abc::IUInt32Property>(userProps, *propHeader, "mvg_viewId", sampleFrame);
        }
        catch(Alembic::Util::Exception&)
        {
          viewId = getAbcProp<Alembic::Abc::IInt32Property>(userProps, *propHeader, "mvg_viewId", sampleFrame);
        }
      }
      if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_poseId"))
      {
        try
        {
          poseId = getAbcProp<Alembic::Abc::IUInt32Property>(userProps, *propHeader, "mvg_poseId", sampleFrame);
        }
        catch(Alembic::Util::Exception&)
        {
          poseId = getAbcProp<Alembic::Abc::IInt32Property>(userProps, *propHeader, "mvg_poseId", sampleFrame);
        }
      }
      if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_intrinsicId"))
      {
        try
        {
          intrinsicId = getAbcProp<Alembic::Abc::IUInt32Property>(userProps, *propHeader, "mvg_intrinsicId", sampleFrame);
        }
        catch(Alembic::Util::Exception&)
        {
          intrinsicId = getAbcProp<Alembic::Abc::IInt32Property>(userProps, *propHeader, "mvg_intrinsicId", sampleFrame);
        }
      }
      if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_rigId"))
      {
        try
        {
          rigId = getAbcProp<Alembic::Abc::IUInt32Property>(userProps, *propHeader, "mvg_rigId", sampleFrame);
        }
        catch(Alembic::Util::Exception&)
        {
          rigId = getAbcProp<Alembic::Abc::IInt32Property>(userProps, *propHeader, "mvg_rigId", sampleFrame);
        }
      }
      if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_subPoseId"))
      {
        try
        {
          subPoseId = getAbcProp<Alembic::Abc::IUInt32Property>(userProps, *propHeader, "mvg_subPoseId", sampleFrame);
        }
        catch(Alembic::Util::Exception&)
        {
          subPoseId = getAbcProp<Alembic::Abc::IInt32Property>(userProps, *propHeader, "mvg_subPoseId", sampleFrame);
        }
      }
      if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_resectionId"))
      {
        try
        {
          resectionId = getAbcProp<Alembic::Abc::IUInt32Property>(userProps, *propHeader, "mvg_resectionId", sampleFrame);
        }
        catch(Alembic::Util::Exception&)
        {
          resectionId = getAbcProp<Alembic::Abc::IInt32Property>(userProps, *propHeader, "mvg_resectionId", sampleFrame);
        }
      }
      if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_poseLocked"))
      {
        poseLocked = getAbcProp<Alembic::Abc::IBoolProperty>(userProps, *propHeader, "mvg_poseLocked", sampleFrame);
      }
      if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_poseLocked"))
      {
        intrinsicLocked = getAbcProp<Alembic::Abc::IBoolProperty>(userProps, *propHeader, "mvg_intrinsicLocked", sampleFrame);
      }
      if(userProps.getPropertyHeader("mvg_sensorSizePix"))
      {
        getAbcArrayProp<Alembic::Abc::IStringArrayProperty>(userProps, "mvg_metadata", sampleFrame, rawMetadata);
        assert(rawMetadata.size() % 2 == 0);
      }
      if(userProps.getPropertyHeader("mvg_sensorSizePix"))
      {
        try
        {
          getAbcArrayProp<Alembic::Abc::IUInt32ArrayProperty>(userProps, "mvg_sensorSizePix", sampleFrame, sensorSize_pix);
        }
        catch(Alembic::Util::Exception&)
        {
          getAbcArrayProp<Alembic::Abc::IInt32ArrayProperty>(userProps, "mvg_sensorSizePix", sampleFrame, sensorSize_pix);
        }
        assert(sensorSize_pix.size() == 2);
      }
      if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_intrinsicType"))
      {
        mvg_intrinsicType = getAbcProp<Alembic::Abc::IStringProperty>(userProps, *propHeader, "mvg_intrinsicType", sampleFrame);
      }
      if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_initialFocalLengthPix"))
      {
        initialFocalLengthPix = getAbcProp<Alembic::Abc::IDoubleProperty>(userProps, *propHeader, "mvg_initialFocalLengthPix", sampleFrame);
      }
      if(userProps.getPropertyHeader("mvg_intrinsicParams"))
      {
        Alembic::Abc::IDoubleArrayProperty prop(userProps, "mvg_intrinsicParams");
        std::shared_ptr<DoubleArraySample> sample;
        prop.get(sample, ISampleSelector(sampleFrame));
        mvg_intrinsicParams.assign(sample->get(), sample->get()+sample->size());
      }
    }
  }

  if(flagsPart & sfm::ESfMData::INTRINSICS)
  {
    // get known values from alembic
    // const float haperture_cm = camSample.getHorizontalAperture();
    // const float vaperture_cm = camSample.getVerticalAperture();
    // compute other needed values
    // const float sensorWidth_mm = std::max(vaperture_cm, haperture_cm) * 10.0;
    // const float mm2pix = sensorSize_pix.at(0) / sensorWidth_mm;
    // imgWidth = haperture_cm * 10.0 * mm2pix;
    // imgHeight = vaperture_cm * 10.0 * mm2pix;

    // create intrinsic parameters object
    std::shared_ptr<Pinhole> pinholeIntrinsic = createPinholeIntrinsic(EINTRINSIC_stringToEnum(mvg_intrinsicType));

    pinholeIntrinsic->setWidth(sensorSize_pix.at(0));
    pinholeIntrinsic->setHeight(sensorSize_pix.at(1));
    pinholeIntrinsic->updateFromParams(mvg_intrinsicParams);
    pinholeIntrinsic->setInitialFocalLengthPix(initialFocalLengthPix);

    if(intrinsicLocked)
      pinholeIntrinsic->lock();
    else
      pinholeIntrinsic->unlock();

    sfmData.intrinsics[intrinsicId] = pinholeIntrinsic;
  }

  // add imported data to the SfMData container TODO use UID
  // this view is incomplete if no flag VIEWS
  std::shared_ptr<View> view = std::make_shared<View>(imagePath,
                                                      viewId,
                                                      intrinsicId,
                                                      poseId,
                                                      sensorSize_pix.at(0),
                                                      sensorSize_pix.at(1),
                                                      rigId,
                                                      subPoseId);
  if(flagsPart & sfm::ESfMData::VIEWS)
  {
    view->setResectionId(resectionId);

    // set metadata
    for(std::size_t i = 0; i < rawMetadata.size(); i+=2)
      view->addMetadata(rawMetadata.at(i), rawMetadata.at(i + 1));

    sfmData.views[viewId] = view;
  }

  if(flagsPart & sfm::ESfMData::EXTRINSICS &&
     isReconstructed)
  {
    // camera
    Mat3 camR;
    camR(0,0) = mat[0][0];
    camR(0,1) = mat[0][1];
    camR(0,2) = mat[0][2];
    camR(1,0) = mat[1][0];
    camR(1,1) = mat[1][1];
    camR(1,2) = mat[1][2];
    camR(2,0) = mat[2][0];
    camR(2,1) = mat[2][1];
    camR(2,2) = mat[2][2];

    Vec3 camT;
    camT(0) = mat[3][0];
    camT(1) = mat[3][1];
    camT(2) = mat[3][2];

    // correct camera orientation from alembic
    const Mat3 scale = Vec3(1,-1,-1).asDiagonal();
    camR = scale * camR;

    Pose3 pose(camR, camT);

    if(view->isPartOfRig())
    {
      Rig& rig = sfmData.getRigs().at(view->getRigId());
      RigSubPose& subPose = rig.getSubPose(view->getSubPoseId());
      if(subPose.status == ERigSubPoseStatus::UNINITIALIZED)
      {
        subPose.status = ERigSubPoseStatus::ESTIMATED;
        subPose.pose = pose;
      }
    }
    else
    {
      sfmData.setPose(*view, CameraPose(pose, poseLocked));
    }
  }

  return true;
}

bool readXform(IXform& xform, M44d& mat, sfm::SfMData& sfmData, sfm::ESfMData flagsPart, bool isReconstructed = true)
{
  using namespace aliceVision::geometry;
  using namespace aliceVision::camera;
  using namespace aliceVision::sfm;

  IXformSchema schema = xform.getSchema();
  XformSample xsample;

  schema.get(xsample);

  // If we have an animated camera we handle it with the xform here
  if(xform.getSchema().getNumSamples() != 1)
  {
    ALICEVISION_LOG_DEBUG(xform.getSchema().getNumSamples() << " samples found in this animated xform.");
    for(index_t frame = 0; frame < xform.getSchema().getNumSamples(); ++frame)
    {
      xform.getSchema().get(xsample, ISampleSelector(frame));
      readCamera(ICamera(xform.getChild(0), kWrapExisting) , mat * xsample.getMatrix(), sfmData, flagsPart, frame, isReconstructed);
    }
    return true;
  }

  mat *= xsample.getMatrix();

  if( !(flagsPart & sfm::ESfMData::EXTRINSICS) )
    return true;

  ICompoundProperty userProps = getAbcUserProperties(schema);

  // Check if it is a rig node
  IndexT rigId = UndefinedIndexT;
  IndexT poseId = UndefinedIndexT;
  std::size_t nbSubPoses = 0;
  bool rigPoseLocked = false;

  if(userProps)
  {
    if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_rigId"))
    {
      try
      {
        rigId = getAbcProp<Alembic::Abc::IUInt32Property>(userProps, *propHeader, "mvg_rigId", 0);
      }
      catch(Alembic::Util::Exception&)
      {
        rigId = getAbcProp<Alembic::Abc::IInt32Property>(userProps, *propHeader, "mvg_rigId", 0);
      }
    }

    if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_poseId"))
    {
      try
      {
        poseId = getAbcProp<Alembic::Abc::IUInt32Property>(userProps, *propHeader, "mvg_poseId", 0);
      }
      catch(Alembic::Util::Exception&)
      {
        poseId = getAbcProp<Alembic::Abc::IInt32Property>(userProps, *propHeader, "mvg_poseId", 0);
      }
    }

    if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_nbSubPoses"))
    {
      try
      {
        nbSubPoses = getAbcProp<Alembic::Abc::IUInt16Property>(userProps, *propHeader, "mvg_nbSubPoses", 0);
      }
      catch(Alembic::Util::Exception&)
      {
        nbSubPoses = getAbcProp<Alembic::Abc::IInt16Property>(userProps, *propHeader, "mvg_nbSubPoses", 0);
      }
    }

    if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_rigPoseLocked"))
    {
      rigPoseLocked = getAbcProp<Alembic::Abc::IBoolProperty>(userProps, *propHeader, "mvg_rigPoseLocked", 0);
    }
  }

  if((rigId == UndefinedIndexT) && (poseId == UndefinedIndexT))
  {
    return true; //not a rig
  }

  if(isReconstructed)
  {
    Mat3 matR;
    matR(0,0) = mat[0][0];
    matR(0,1) = mat[0][1];
    matR(0,2) = mat[0][2];
    matR(1,0) = mat[1][0];
    matR(1,1) = mat[1][1];
    matR(1,2) = mat[1][2];
    matR(2,0) = mat[2][0];
    matR(2,1) = mat[2][1];
    matR(2,2) = mat[2][2];

    Vec3 matT;
    matT(0) = mat[3][0];
    matT(1) = mat[3][1];
    matT(2) = mat[3][2];

    Pose3 pose(matR, matT);

    if(sfmData.getPoses().find(poseId) == sfmData.getPoses().end())
      sfmData.getPoses().emplace(poseId, CameraPose(pose, rigPoseLocked));
  }

  if(sfmData.getRigs().find(rigId) == sfmData.getRigs().end())
    sfmData.getRigs().emplace(rigId, Rig(nbSubPoses));

  mat.makeIdentity();
  return true;
}

// Top down read of 3d objects
void visitObject(IObject iObj, M44d mat, sfm::SfMData& sfmdata, sfm::ESfMData flagsPart, bool isReconstructed = true)
{
  // ALICEVISION_LOG_DEBUG("ABC visit: " << iObj.getFullName());
  if(iObj.getName() == "mvgCamerasUndefined")
    isReconstructed = false;
  
  const MetaData& md = iObj.getMetaData();
  if(IPoints::matches(md) && (flagsPart & sfm::ESfMData::STRUCTURE))
  {
    readPointCloud(iObj, mat, sfmdata, flagsPart);
  }
  else if(IXform::matches(md))
  {
    IXform xform(iObj, kWrapExisting);
    readXform(xform, mat, sfmdata, flagsPart, isReconstructed);
  }
  else if(ICamera::matches(md) && ((flagsPart & sfm::ESfMData::VIEWS) ||
                                   (flagsPart & sfm::ESfMData::INTRINSICS) ||
                                   (flagsPart & sfm::ESfMData::EXTRINSICS)))
  {
    ICamera check_cam(iObj, kWrapExisting);
    // If it's not an animated camera we add it here
    if(check_cam.getSchema().getNumSamples() == 1)
    {
      readCamera(check_cam, mat, sfmdata, flagsPart, 0, isReconstructed);
    }
  }

  // Recurse
  for(std::size_t i = 0; i < iObj.getNumChildren(); i++)
  {
    visitObject(iObj.getChild(i), mat, sfmdata, flagsPart, isReconstructed);
  }
}

struct AlembicImporter::DataImpl
{
  DataImpl(const std::string& filename)
  {
    Alembic::AbcCoreFactory::IFactory factory;
    Alembic::AbcCoreFactory::IFactory::CoreType coreType;
    Abc::IArchive archive = factory.getArchive(filename, coreType);

    if(!archive.valid())
      throw std::runtime_error("Can't open '" + filename + "' : Alembic file is not valid.");

    _rootEntity = archive.getTop();
    _filename = filename;
  }
  
  IObject _rootEntity;
  std::string _filename;
};

AlembicImporter::AlembicImporter(const std::string& filename)
{
  _dataImpl.reset(new DataImpl(filename));
}

AlembicImporter::~AlembicImporter()
{}

void AlembicImporter::populateSfM(sfm::SfMData& sfmdata, sfm::ESfMData flagsPart)
{
  const index_t sampleFrame = 0;
  IObject rootObj = _dataImpl->_rootEntity.getChild("mvgRoot");
  ICompoundProperty userProps = rootObj.getProperties();

  // set SfMData folder absolute path
  sfmdata.setAbsolutePath(_dataImpl->_filename);

  if(userProps.getPropertyHeader("mvg_featuresFolders"))
  {
    std::vector<std::string> featuresFolders;
    getAbcArrayProp<Alembic::Abc::IStringArrayProperty>(userProps, "mvg_featuresFolders", sampleFrame, featuresFolders);
    sfmdata.setFeaturesFolders(featuresFolders);
  }

  if(userProps.getPropertyHeader("mvg_matchesFolders"))
  {
    std::vector<std::string> matchesFolders;
    getAbcArrayProp<Alembic::Abc::IStringArrayProperty>(userProps, "mvg_matchesFolders", sampleFrame, matchesFolders);
    sfmdata.setMatchesFolders(matchesFolders);
  }

  // keep compatibility with single folder for feature and matching
  if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_featureFolder"))
  {
    const std::string featuresFolder = getAbcProp<Alembic::Abc::IStringProperty>(userProps, *propHeader, "mvg_featureFolder", sampleFrame);
    sfmdata.addFeaturesFolder(featuresFolder);
  }

  if(const Alembic::Abc::PropertyHeader *propHeader = userProps.getPropertyHeader("mvg_matchingFolder"))
  {
    const std::string matchesFolder = getAbcProp<Alembic::Abc::IStringProperty>(userProps, *propHeader, "mvg_matchingFolder", sampleFrame);
    sfmdata.addMatchesFolder(matchesFolder);
  }

  // TODO : handle the case where the archive wasn't correctly opened
  M44d xformMat;
  visitObject(_dataImpl->_rootEntity, xformMat, sfmdata, flagsPart);

  // TODO: fusion of common intrinsics
}

} // namespace sfm
} // namespace aliceVision
