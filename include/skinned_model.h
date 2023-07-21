#pragma once

#include <map>
#include <vector>
#include <glad/glad.h>
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h> 
#include <assimp/postprocess.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "texture.h"
#include "assimp_helpers.h"
#include "material.h"

#define ASSIMP_LOAD_FLAGS (aiProcess_Triangulate | aiProcess_GenSmoothNormals |  aiProcess_JoinIdenticalVertices )
#define MAX_NUM_BONES_PER_VERTEX 6
#define POSITION_LOCATION    0
#define TEX_COORD_LOCATION   1
#define NORMAL_LOCATION      2
#define BONE_ID_LOCATION0     3
#define BONE_ID_LOCATION1     4
#define BONE_WEIGHT_LOCATION0 5
#define BONE_WEIGHT_LOCATION1 6

class SkinnedModel
{
public:
    SkinnedModel(const std::string& Filename) 
    {
        LoadMesh(Filename);
        leap_bone_map = 
        {
            {0, "Wrist"},
            {2, "thumb_meta"},
            {3, "thumb_a"},
            {4, "thumb_b"},
            {5, "thumb_end"},
            {6, "index_meta"},
            {7, "index_a"},
            {8, "index_b"},
            {9, "index_c"},
            {10, "middle_meta"},
            {11, "middle_a"},
            {12, "middle_b"},
            {13, "middle_c"},
            {14, "ring_meta"},
            {15, "ring_a"},
            {16, "ring_b"},
            {17, "ring_c"},
            {18, "pinky_meta"},
            {19, "pinky_a"},
            {20, "pinky_b"},
            {21, "pinky_c"}
        };
    };

    ~SkinnedModel();

    bool LoadMesh(const std::string& Filename);

    void Render();

    unsigned int NumBones() const
    {
        return (unsigned int)m_BoneNameToIndexMap.size();
    }

    // WorldTrans& GetWorldTransform() { return m_worldTransform; }

    const Material& GetMaterial();

    void GetBoneTransforms(float AnimationTimeSec, std::vector<glm::mat4>& Transforms, const std::vector<glm::mat4> leap_bone_transforms);

private:
    

    void Clear();

    bool InitFromScene(const aiScene* pScene, const std::string& Filename);

    void CountVerticesAndIndices(const aiScene* pScene, unsigned int& NumVertices, unsigned int& NumIndices);

    void ReserveSpace(unsigned int NumVertices, unsigned int NumIndices);

    void InitAllMeshes(const aiScene* pScene);

    void InitSingleMesh(unsigned int MeshIndex, const aiMesh* paiMesh);

    bool InitMaterials(const aiScene* pScene, const std::string& Filename);

    void PopulateBuffers();

    void LoadTextures(const std::string& Dir, const aiMaterial* pMaterial, int index);

    void LoadDiffuseTexture(const std::string& Dir, const aiMaterial* pMaterial, int index);

    void LoadSpecularTexture(const std::string& Dir, const aiMaterial* pMaterial, int index);

    void LoadColors(const aiMaterial* pMaterial, int index);

    struct VertexBoneData
    {
        unsigned int BoneIDs[MAX_NUM_BONES_PER_VERTEX] = { 0 };
        float Weights[MAX_NUM_BONES_PER_VERTEX] = { 0.0f };

        VertexBoneData()
        {
        }

        void AddBoneData(unsigned int BoneID, float Weight)
        {
            for (unsigned int i = 0 ; i < sizeof(BoneIDs)/sizeof(BoneIDs[0]) ; i++) {
                if (Weights[i] == 0.0) {
                    BoneIDs[i] = BoneID;
                    Weights[i] = Weight;
                    //printf("Adding bone %d weight %f at index %i\n", BoneID, Weight, i);
                    return;
                }
            }

            // should never get here - more bones than we have space for
            assert(0);
        }
    };

    void LoadMeshBones(unsigned int MeshIndex, const aiMesh* paiMesh);
    void LoadSingleBone(unsigned int MeshIndex, const aiBone* pBone);
    int GetBoneId(const aiBone* pBone);
    void CalcInterpolatedScaling(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);
    void CalcInterpolatedRotation(aiQuaternion& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);
    void CalcInterpolatedPosition(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);
    unsigned int FindScaling(float AnimationTime, const aiNodeAnim* pNodeAnim);
    unsigned int FindRotation(float AnimationTime, const aiNodeAnim* pNodeAnim);
    unsigned int FindPosition(float AnimationTime, const aiNodeAnim* pNodeAnim);
    const aiNodeAnim* FindNodeAnim(const aiAnimation* pAnimation, const std::string& NodeName);
    std::string GetDirFromFilename(const std::string& Filename);
    // void ReadNodeHierarchy(float AnimationTime, const aiNode* pNode, const glm::mat4& ParentTransform);

#define INVALID_MATERIAL 0xFFFFFFFF

    enum BUFFER_TYPE {
        INDEX_BUFFER = 0,
        POS_VB       = 1,
        TEXCOORD_VB  = 2,
        NORMAL_VB    = 3,
        BONE_VB      = 4,
        NUM_BUFFERS  = 5
    };

    // WorldTrans m_worldTransform;
    unsigned int m_VAO = 0;
    unsigned int m_Buffers[NUM_BUFFERS] = { 0 };

    struct BasicMeshEntry {
        BasicMeshEntry()
        {
            NumIndices = 0;
            BaseVertex = 0;
            BaseIndex = 0;
            MaterialIndex = INVALID_MATERIAL;
        }

        unsigned int NumIndices;
        unsigned int BaseVertex;
        unsigned int BaseIndex;
        unsigned int MaterialIndex;
    };

    Assimp::Importer Importer;
    const aiScene* pScene = NULL;
    std::vector<BasicMeshEntry> m_Meshes;
    std::vector<Material> m_Materials;

    // Temporary space for vertex stuff before we load them into the GPU
    std::vector<glm::vec3> m_Positions;
    std::vector<glm::vec3> m_Normals;
    std::vector<glm::vec2> m_TexCoords;
    std::vector<unsigned int> m_Indices;
    std::vector<VertexBoneData> m_Bones;

    std::map<std::string, unsigned int> m_BoneNameToIndexMap;
    std::map<unsigned int, std::string> leap_bone_map;
    struct BoneInfo
    {
        glm::mat4 OffsetMatrix;
        glm::mat4 FinalTransformation;

        BoneInfo(const glm::mat4& Offset)
        {
            OffsetMatrix = Offset;
            FinalTransformation = glm::mat4();
        }
    };

    std::vector<BoneInfo> m_BoneInfo;
    glm::mat4 m_GlobalInverseTransform;
};
