#ifndef SKINNED_MODEL_H
#define SKINNED_MODEL_H

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
#include "GLMhelpers.h"
#include "material.h"
#include "skinned_shader.h"
#include "fbo.h"

#define INVALID_MATERIAL 0xFFFFFFFF
#define ASSIMP_LOAD_FLAGS (aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices)
#define MAX_NUM_BONES_PER_VERTEX 6
#define POSITION_LOCATION 0
#define VERTEX_COLOR_LOCATION 1
#define TEX_COORD_LOCATION 2
#define NORMAL_LOCATION 3
#define BONE_ID_LOCATION0 4
#define BONE_ID_LOCATION1 5
#define BONE_WEIGHT_LOCATION0 6
#define BONE_WEIGHT_LOCATION1 7

enum BUFFER_TYPE
{
    INDEX_BUFFER = 0,
    POS_VB = 1,
    TEXCOORD_VB = 2,
    NORMAL_VB = 3,
    BONE_VB = 4,
    VERTEX_COLOR_VB = 5,
    NUM_BUFFERS = 6
};

struct BasicMeshEntry
{
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

struct VertexBoneData
{
    unsigned int BoneIDs[MAX_NUM_BONES_PER_VERTEX] = {0};
    float Weights[MAX_NUM_BONES_PER_VERTEX] = {0.0f};

    VertexBoneData()
    {
    }

    void AddBoneData(unsigned int BoneID, float Weight)
    {
        for (unsigned int i = 0; i < sizeof(BoneIDs) / sizeof(BoneIDs[0]); i++)
        {
            if (Weights[i] == 0.0)
            {
                BoneIDs[i] = BoneID;
                Weights[i] = Weight;
                // printf("Adding bone %d weight %f at index %i\n", BoneID, Weight, i);
                return;
            }
        }

        // should never get here - more bones than we have space for
        assert(0);
    }
    float sum_weights()
    {
        float sum = 0.0f;
        for (unsigned int i = 0; i < MAX_NUM_BONES_PER_VERTEX; i++)
        {
            sum += Weights[i];
        }
        return sum;
    }
};

struct BoneInfo
{
    glm::mat4 LocalToBoneTransform; // from local space to bone space
    glm::mat4 FinalTransformation;  // from bone space to root space

    BoneInfo(const glm::mat4 &l2b_transform)
    {
        LocalToBoneTransform = l2b_transform;
        FinalTransformation = glm::mat4();
    }
};

class SkinnedModel
{
public:
    SkinnedModel(const std::string &Filename, const std::string &ExternalTextureFileName = "",
                 const unsigned int proj_width = 0, const unsigned int proj_height = 0,
                 const unsigned int cam_width = 0, const unsigned int cam_height = 0,
                 bool left_chirality = true) : m_fbo(proj_width, proj_height)
    {
        m_externalTextureFileName = ExternalTextureFileName;
        m_width = proj_width;
        m_height = proj_height;
        m_camHeight = cam_height;
        m_camWidth = cam_width;
        m_leftChirality = left_chirality;
        bool success = LoadMesh(Filename);
        if (!success)
        {
            std::cout << "Error loading mesh\n"
                      << std::endl;
            exit(1);
        }
    };
    ~SkinnedModel()
    {
        Clear();
    };
    bool LoadMesh(const std::string &Filename);
    void Render(Shader &shader, unsigned int camTex, bool useFBO);
    void Render(SkinningShader &shader, const std::vector<glm::mat4> &bones_to_world,
                const glm::mat4 &local_to_world, bool use_bones = false);
    void Render(SkinningShader &shader, const std::vector<glm::mat4> &bones_to_world,
                const glm::mat4 &local_to_world, unsigned int camTex, bool useFBO = true, bool use_bones = false);
    const Material &GetMaterial();
    void GetBoneTransforms(std::vector<glm::mat4> &transforms, const std::vector<glm::mat4> leap_bone_transforms, const glm::mat4 local_to_world, const bool use_bones = false);
    void GetBoneTransformsHack(std::vector<glm::mat4> &transforms, const std::vector<glm::mat4> bones_to_world);
    glm::vec3 getCenterOfMass();
    std::string getBoneName(unsigned int index);
    unsigned int NumBones() const
    {
        return (unsigned int)m_BoneNameToIndexMap.size();
    };
    void GetLocalToBoneTransforms(std::vector<glm::mat4> &transforms, bool inverse = false, bool only_leap_bones = false);
    void GetBoneFinalTransforms(std::vector<glm::mat4> &transforms);
    void GetBoneTransformRelativeToParent(std::vector<glm::mat4> &transforms);
    FBO m_fbo;

private:
    void Clear();
    bool InitFromScene(const aiScene *pScene, const std::string &Filename);
    void CountVerticesAndIndices(const aiScene *pScene, unsigned int &NumVertices, unsigned int &NumIndices);
    void ReserveSpace(unsigned int NumVertices, unsigned int NumIndices);
    void InitAllMeshes(const aiScene *pScene);
    void InitSingleMesh(unsigned int MeshIndex, const aiMesh *paiMesh);
    bool InitMaterials(const aiScene *pScene, const std::string &Filename);
    void PopulateBuffers();
    void LoadTextures(const std::string &Dir, const aiMaterial *pMaterial, int index);
    void LoadDiffuseTexture(const std::string &Dir, const aiMaterial *pMaterial, int index);
    void LoadSpecularTexture(const std::string &Dir, const aiMaterial *pMaterial, int index);
    void LoadColors(const aiMaterial *pMaterial, int index);
    void LoadMeshBones(unsigned int MeshIndex, const aiMesh *paiMesh);
    void LoadSingleBone(unsigned int MeshIndex, const aiBone *pBone);
    int GetBoneId(const aiBone *pBone);
    void CalcInterpolatedScaling(aiVector3D &Out, float AnimationTime, const aiNodeAnim *pNodeAnim);
    void CalcInterpolatedRotation(aiQuaternion &Out, float AnimationTime, const aiNodeAnim *pNodeAnim);
    void CalcInterpolatedPosition(aiVector3D &Out, float AnimationTime, const aiNodeAnim *pNodeAnim);
    unsigned int FindScaling(float AnimationTime, const aiNodeAnim *pNodeAnim);
    unsigned int FindRotation(float AnimationTime, const aiNodeAnim *pNodeAnim);
    unsigned int FindPosition(float AnimationTime, const aiNodeAnim *pNodeAnim);
    void ReadNodeHierarchy(const aiNode *pNode, const glm::mat4 &ParentTransform);
    std::string GetDirFromFilename(const std::string &Filename);

    unsigned int m_width, m_height;
    bool m_leftChirality;
    unsigned int m_camWidth, m_camHeight;
    unsigned int m_VAO = 0;
    unsigned int m_Buffers[NUM_BUFFERS] = {0};
    // unsigned int m_FBO, m_fbo_depth_buffer, m_fbo_texture;
    // unsigned int m_cam_texture;

    Assimp::Importer Importer;
    const aiScene *pScene = NULL;
    std::string m_externalTextureFileName;
    std::vector<BasicMeshEntry> m_Meshes;
    std::vector<Material> m_Materials;

    // Temporary space for vertex stuff before we load them into the GPU
    std::vector<glm::vec3> m_Positions;
    std::vector<glm::vec3> m_Normals;
    std::vector<glm::vec2> m_TexCoords;
    std::vector<glm::vec3> m_VertColors;
    std::vector<unsigned int> m_Indices;
    std::vector<VertexBoneData> m_Bones; // per vertex bone data

    std::map<std::string, unsigned int> m_BoneNameToIndexMap; // maps a bone name in loaded mesh to its index
    std::map<unsigned int, std::string> m_BoneIndexToNameMap; // inverse of above
    std::map<std::string, unsigned int> bone_leap_map;        // maps bone names from the map above to their leapmotion index

    std::vector<BoneInfo> m_BoneInfo; // per bone info
    glm::mat4 m_GlobalInverseTransform;
};
#endif