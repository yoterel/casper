/*

        Copyright 2011 Etay Meiri

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "skinned_model.h"
#include "assimp_helpers.h"

void SkinnedModel::Clear()
{
    if (m_Buffers[0] != 0) {
        glDeleteBuffers(sizeof(m_Buffers)/sizeof(m_Buffers[0]), m_Buffers);
    }

    if (m_VAO != 0) {
        glDeleteVertexArrays(1, &m_VAO);
        m_VAO = 0;
    }
}


bool SkinnedModel::LoadMesh(const std::string& Filename)
{
    // Release the previously loaded mesh (if it exists)
    Clear();

    // Create the VAO
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    // Create the buffers for the vertices attributes
    glGenBuffers(sizeof(m_Buffers)/sizeof(m_Buffers[0]), m_Buffers);

    bool Ret = false;

    pScene = Importer.ReadFile(Filename.c_str(), ASSIMP_LOAD_FLAGS);

    if (pScene) {
        m_GlobalInverseTransform = AssimpGLMHelpers::ConvertMatrixToGLMFormat(pScene->mRootNode->mTransformation);
        m_GlobalInverseTransform = glm::inverse(m_GlobalInverseTransform);
        Ret = InitFromScene(pScene, Filename);
    }
    else {
        std::cout << "Error parsing '" << Filename << "': '" << Importer.GetErrorString() << "'" << std::endl;
    }

    // Make sure the VAO is not changed from the outside
    glBindVertexArray(0);
    bone_leap_map = 
        {
            // {"Elbow", 0},
            {"thumb_meta", 3},
            {"thumb_a", 4},
            {"thumb_b", 5},
            
            {"index_meta", 6},
            {"index_a", 7},
            {"index_b", 8},
            {"index_c", 9},

            {"middle_meta", 10},
            {"middle_a", 11},
            {"middle_b", 12},
            {"middle_c", 13},

            {"ring_meta", 14},
            {"ring_a", 15},
            {"ring_b", 16},
            {"ring_c", 17},

            {"pinky_meta", 18},
            {"pinky_a", 19},
            {"pinky_b", 20},
            {"pinky_c", 21}
        };
    return Ret;
}

bool SkinnedModel::InitFromScene(const aiScene* pScene, const std::string& Filename)
{
    m_Meshes.resize(pScene->mNumMeshes);
    m_Materials.resize(pScene->mNumMaterials);

    unsigned int NumVertices = 0;
    unsigned int NumIndices = 0;

    CountVerticesAndIndices(pScene, NumVertices, NumIndices);

    ReserveSpace(NumVertices, NumIndices);

    InitAllMeshes(pScene);

    if (!InitMaterials(pScene, Filename)) {
        return false;
    }

    PopulateBuffers();

    return glGetError() == GL_NO_ERROR;
}

void SkinnedModel::CountVerticesAndIndices(const aiScene* pScene, unsigned int& NumVertices, unsigned int& NumIndices)
{
    for (unsigned int i = 0 ; i < m_Meshes.size() ; i++) {
        m_Meshes[i].MaterialIndex = pScene->mMeshes[i]->mMaterialIndex;
        m_Meshes[i].NumIndices = pScene->mMeshes[i]->mNumFaces * 3;
        m_Meshes[i].BaseVertex = NumVertices;
        m_Meshes[i].BaseIndex = NumIndices;

        NumVertices += pScene->mMeshes[i]->mNumVertices;
        NumIndices  += m_Meshes[i].NumIndices;
    }
}

void SkinnedModel::ReserveSpace(unsigned int NumVertices, unsigned int NumIndices)
{
    m_Positions.reserve(NumVertices);
    m_Normals.reserve(NumVertices);
    m_TexCoords.reserve(NumVertices);
    m_Indices.reserve(NumIndices);
    m_Bones.resize(NumVertices);
}

void SkinnedModel::InitAllMeshes(const aiScene* pScene)
{
    for (unsigned int i = 0 ; i < m_Meshes.size() ; i++) {
        const aiMesh* paiMesh = pScene->mMeshes[i];
        InitSingleMesh(i, paiMesh);
    }
}

void SkinnedModel::InitSingleMesh(unsigned int MeshIndex, const aiMesh* paiMesh)
{
    const aiVector3D Zero3D(0.0f, 0.0f, 0.0f);

    // Populate the vertex attribute vectors
    for (unsigned int i = 0 ; i < paiMesh->mNumVertices ; i++) {

        const aiVector3D& pPos      = paiMesh->mVertices[i];
        m_Positions.push_back(glm::vec3(pPos.x, pPos.y, pPos.z));

        if (paiMesh->mNormals) {
            const aiVector3D& pNormal   = paiMesh->mNormals[i];
            m_Normals.push_back(glm::vec3(pNormal.x, pNormal.y, pNormal.z));
        } else {
            aiVector3D Normal(0.0f, 1.0f, 0.0f);
            m_Normals.push_back(glm::vec3(Normal.x, Normal.y, Normal.z));
        }

        const aiVector3D& pTexCoord = paiMesh->HasTextureCoords(0) ? paiMesh->mTextureCoords[0][i] : Zero3D;
        m_TexCoords.push_back(glm::vec2(pTexCoord.x, pTexCoord.y));
    }

    LoadMeshBones(MeshIndex, paiMesh);

    // Populate the index buffer
    for (unsigned int i = 0 ; i < paiMesh->mNumFaces ; i++) {
        const aiFace& Face = paiMesh->mFaces[i];
        //        printf("num indices %d\n", Face.mNumIndices);
        //        assert(Face.mNumIndices == 3);
        m_Indices.push_back(Face.mIndices[0]);
        m_Indices.push_back(Face.mIndices[1]);
        m_Indices.push_back(Face.mIndices[2]);
    }
}

void SkinnedModel::LoadMeshBones(unsigned int MeshIndex, const aiMesh* pMesh)
{
    for (unsigned int i = 0 ; i < pMesh->mNumBones ; i++) {
        LoadSingleBone(MeshIndex, pMesh->mBones[i]);
    }
    for (unsigned int i=0; i < m_Bones.size() ; i++) {
        float sum_weight = m_Bones[i].sum_weights();
        if (abs(sum_weight - 1.0f) > 0.01f)
        {
            std::cout << "vertex " << i << " has sum of weights " << sum_weight << std::endl;
            assert(false);
        }
    }
    for (std::map<std::string, unsigned int>::iterator i = m_BoneNameToIndexMap.begin(); i != m_BoneNameToIndexMap.end(); ++i)
        m_BoneIndexToNameMap[i->second] = i->first;
    glm::mat4 iden = glm::mat4(1.0f);
    ReadNodeHierarchy(pScene->mRootNode, iden);
}

std::string SkinnedModel::getBoneName(unsigned int index)
{
    if (index >= m_BoneIndexToNameMap.size())
        return "";
    return m_BoneIndexToNameMap[index];
}

void SkinnedModel::LoadSingleBone(unsigned int MeshIndex, const aiBone* pBone)
{
    int BoneId = GetBoneId(pBone);

    if (BoneId == m_BoneInfo.size()) {
        BoneInfo bi(AssimpGLMHelpers::ConvertMatrixToGLMFormat(pBone->mOffsetMatrix));
        m_BoneInfo.push_back(bi);
    }

    for (unsigned int i = 0 ; i < pBone->mNumWeights ; i++) {
        const aiVertexWeight& vw = pBone->mWeights[i];
        unsigned int GlobalVertexID = m_Meshes[MeshIndex].BaseVertex + pBone->mWeights[i].mVertexId;
        m_Bones[GlobalVertexID].AddBoneData(BoneId, vw.mWeight);
    }
}

int SkinnedModel::GetBoneId(const aiBone* pBone)
{
    int BoneIndex = 0;
    std::string BoneName(pBone->mName.C_Str());

    if (m_BoneNameToIndexMap.find(BoneName) == m_BoneNameToIndexMap.end()) {
        // Allocate an index for a new bone
        BoneIndex = (int)m_BoneNameToIndexMap.size();
        m_BoneNameToIndexMap[BoneName] = BoneIndex;
    }
    else {
        BoneIndex = m_BoneNameToIndexMap[BoneName];
    }

    return BoneIndex;
}

bool SkinnedModel::InitMaterials(const aiScene* pScene, const std::string& Filename)
{
    std::string Dir = GetDirFromFilename(Filename);

    bool Ret = true;

    printf("Num materials: %d\n", pScene->mNumMaterials);

    // Initialize the materials
    for (unsigned int i = 0 ; i < pScene->mNumMaterials ; i++) {
        const aiMaterial* pMaterial = pScene->mMaterials[i];

        LoadTextures(Dir, pMaterial, i);

        LoadColors(pMaterial, i);
    }

    return Ret;
}

void SkinnedModel::LoadTextures(const std::string& Dir, const aiMaterial* pMaterial, int index)
{
    LoadDiffuseTexture(Dir, pMaterial, index);
    LoadSpecularTexture(Dir, pMaterial, index);
}

void SkinnedModel::LoadDiffuseTexture(const std::string& Dir, const aiMaterial* pMaterial, int index)
{
    m_Materials[index].pDiffuse = NULL;

    if (pMaterial->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
        aiString Path;

        if (pMaterial->GetTexture(aiTextureType_DIFFUSE, 0, &Path, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS) {
            std::string p(Path.data);

            if (p.substr(0, 2) == ".\\") {
                p = p.substr(2, p.size() - 2);
            }

            std::string FullPath = Dir + "/" + p;

            m_Materials[index].pDiffuse = new Texture(GL_TEXTURE_2D, FullPath.c_str());

            if (!m_Materials[index].pDiffuse->Load()) {
                printf("Error loading diffuse texture '%s'\n", FullPath.c_str());
                exit(0);
            }
            else {
                printf("Loaded diffuse texture '%s'\n", FullPath.c_str());
            }
        }
    }
}

void SkinnedModel::LoadSpecularTexture(const std::string& Dir, const aiMaterial* pMaterial, int index)
{
    m_Materials[index].pSpecularExponent = NULL;

    if (pMaterial->GetTextureCount(aiTextureType_SHININESS) > 0) {
        aiString Path;

        if (pMaterial->GetTexture(aiTextureType_SHININESS, 0, &Path, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS) {
            std::string p(Path.data);

            if (p == "C:\\\\") {
                p = "";
            } else if (p.substr(0, 2) == ".\\") {
                p = p.substr(2, p.size() - 2);
            }

            std::string FullPath = Dir + "/" + p;

            m_Materials[index].pSpecularExponent = new Texture(GL_TEXTURE_2D, FullPath.c_str());

            if (!m_Materials[index].pSpecularExponent->Load()) {
                printf("Error loading specular texture '%s'\n", FullPath.c_str());
                exit(0);
            }
            else {
                printf("Loaded specular texture '%s'\n", FullPath.c_str());
            }
        }
    }
}

void SkinnedModel::LoadColors(const aiMaterial* pMaterial, int index)
{
    aiColor3D AmbientColor(0.0f, 0.0f, 0.0f);
    glm::vec3 AllOnes(1.0f, 1.0f, 1.0f);

    int ShadingModel = 0;
    if (pMaterial->Get(AI_MATKEY_SHADING_MODEL, ShadingModel) == AI_SUCCESS) {
        printf("Shading model %d\n", ShadingModel);
    }

    if (pMaterial->Get(AI_MATKEY_COLOR_AMBIENT, AmbientColor) == AI_SUCCESS) {
        printf("Loaded ambient color [%f %f %f]\n", AmbientColor.r, AmbientColor.g, AmbientColor.b);
        m_Materials[index].AmbientColor.r = AmbientColor.r;
        m_Materials[index].AmbientColor.g = AmbientColor.g;
        m_Materials[index].AmbientColor.b = AmbientColor.b;
    } else {
        m_Materials[index].AmbientColor = AllOnes;
    }

    aiColor3D DiffuseColor(0.0f, 0.0f, 0.0f);

    if (pMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, DiffuseColor) == AI_SUCCESS) {
        printf("Loaded diffuse color [%f %f %f]\n", DiffuseColor.r, DiffuseColor.g, DiffuseColor.b);
        m_Materials[index].DiffuseColor.r = DiffuseColor.r;
        m_Materials[index].DiffuseColor.g = DiffuseColor.g;
        m_Materials[index].DiffuseColor.b = DiffuseColor.b;
    }

    aiColor3D SpecularColor(0.0f, 0.0f, 0.0f);

    if (pMaterial->Get(AI_MATKEY_COLOR_SPECULAR, SpecularColor) == AI_SUCCESS) {
        printf("Loaded specular color [%f %f %f]\n", SpecularColor.r, SpecularColor.g, SpecularColor.b);
        m_Materials[index].SpecularColor.r = SpecularColor.r;
        m_Materials[index].SpecularColor.g = SpecularColor.g;
        m_Materials[index].SpecularColor.b = SpecularColor.b;
    }
}

void SkinnedModel::PopulateBuffers()
{
    glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[POS_VB]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(m_Positions[0]) * m_Positions.size(), &m_Positions[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(POSITION_LOCATION);
    glVertexAttribPointer(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[TEXCOORD_VB]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(m_TexCoords[0]) * m_TexCoords.size(), &m_TexCoords[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(TEX_COORD_LOCATION);
    glVertexAttribPointer(TEX_COORD_LOCATION, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[NORMAL_VB]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(m_Normals[0]) * m_Normals.size(), &m_Normals[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(NORMAL_LOCATION);
    glVertexAttribPointer(NORMAL_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[BONE_VB]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(m_Bones[0]) * m_Bones.size(), &m_Bones[0], GL_STATIC_DRAW);

    glEnableVertexAttribArray(BONE_ID_LOCATION0);
    glVertexAttribIPointer(BONE_ID_LOCATION0, MAX_NUM_BONES_PER_VERTEX-2, GL_INT, sizeof(VertexBoneData), (const GLvoid*)0);
    
    glEnableVertexAttribArray(BONE_ID_LOCATION1);
    glVertexAttribIPointer(BONE_ID_LOCATION1, MAX_NUM_BONES_PER_VERTEX-4, GL_INT, sizeof(VertexBoneData), (const GLvoid*)((MAX_NUM_BONES_PER_VERTEX-2) * sizeof(unsigned int)));
    
    glEnableVertexAttribArray(BONE_WEIGHT_LOCATION0);
    glVertexAttribPointer(BONE_WEIGHT_LOCATION0, MAX_NUM_BONES_PER_VERTEX-2, GL_FLOAT, GL_FALSE, sizeof(VertexBoneData),
                          (const GLvoid*)(MAX_NUM_BONES_PER_VERTEX * sizeof(unsigned int)));

    glEnableVertexAttribArray(BONE_WEIGHT_LOCATION1);
    size_t offset = ((MAX_NUM_BONES_PER_VERTEX) * sizeof(unsigned int)) + ((MAX_NUM_BONES_PER_VERTEX-2) * sizeof(float));
    glVertexAttribPointer(BONE_WEIGHT_LOCATION1, MAX_NUM_BONES_PER_VERTEX-4, GL_FLOAT, GL_FALSE, sizeof(VertexBoneData),
                          (const GLvoid*)offset);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Buffers[INDEX_BUFFER]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_Indices[0]) * m_Indices.size(), &m_Indices[0], GL_STATIC_DRAW);
}

void SkinnedModel::Render(SkinningShader& shader, const std::vector<glm::mat4>& bones_basis, const float animationTime)
{
    shader.use();
    std::vector<glm::mat4> Transforms;
    this->GetBoneTransforms(animationTime, Transforms, bones_basis);
    for (unsigned int i = 0 ; i < Transforms.size() ; i++) {
        shader.SetBoneTransform(i, Transforms[i]);
    }
    glBindVertexArray(m_VAO);

    for (unsigned int i = 0 ; i < m_Meshes.size() ; i++) {
        unsigned int MaterialIndex = m_Meshes[i].MaterialIndex;

        assert(MaterialIndex < m_Materials.size());

        if (m_Materials[MaterialIndex].pDiffuse) {
            m_Materials[MaterialIndex].pDiffuse->Bind(GL_TEXTURE0);
        }

        if (m_Materials[MaterialIndex].pSpecularExponent) {
            m_Materials[MaterialIndex].pSpecularExponent->Bind(GL_TEXTURE6);
        }

        glDrawElementsBaseVertex(GL_TRIANGLES,
                                 m_Meshes[i].NumIndices,
                                 GL_UNSIGNED_INT,
                                 (void*)(sizeof(unsigned int) * m_Meshes[i].BaseIndex),
                                 m_Meshes[i].BaseVertex);
    }

    // Make sure the VAO is not changed from the outside
    glBindVertexArray(0);
}


const Material& SkinnedModel::GetMaterial()
{
    for (unsigned int i = 0 ; i < m_Materials.size() ; i++) {
        if (m_Materials[i].AmbientColor != glm::vec3(0.0f, 0.0f, 0.0f)) {
            return m_Materials[i];
        }
    }

    return m_Materials[0];
}

glm::vec3 SkinnedModel::getCenterOfMass()
{
    glm::vec3 center_of_mass = glm::vec3(0.0f, 0.0f, 0.0f);
    for (unsigned int i = 0 ; i < m_Positions.size() ; i++) {
        center_of_mass += m_Positions[i];
    }
    center_of_mass /= (float)m_Positions.size();
    return center_of_mass;
};

void SkinnedModel::GetBoneTransforms(float AnimationTimeSec, std::vector<glm::mat4>& Transforms, const std::vector<glm::mat4> leap_bone_transforms)
{
    Transforms.resize(m_BoneInfo.size());
    for (unsigned int i = 0 ; i < m_BoneInfo.size() ; i++) {
        Transforms[i] = m_BoneInfo[i].FinalTransformation;
    }
    
    // for (unsigned int i = 0 ; i < m_BoneInfo.size() ; i++) {
    //         // Transforms[i] = m_BoneInfo[i].FinalTransformation;
    //         // if (i == 0)
    //             // test.m[0][0] = sin(TimeInSeconds);
    //         Transforms[i] = iden;
    // }
    if (leap_bone_transforms.size() > 0)
    {
        // glm::mat4 rot = glm::rotate(glm::mat4(1.0f), sin(AnimationTimeSec), glm::vec3(0.0f, 1.0f, 0.0f));
        for (auto const& x : bone_leap_map)
        {
            // if (x.first == "index_b")
            // {
            unsigned int bone_index = m_BoneNameToIndexMap[x.first];
                
            Transforms[bone_index] = glm::inverse(leap_bone_transforms[0]) * leap_bone_transforms[x.second] * m_BoneInfo[bone_index].OffsetMatrix;
            // }
        }
    }
}

void SkinnedModel::ReadNodeHierarchy(const aiNode* pNode, const glm::mat4& ParentTransform)
{
    std::string NodeName(pNode->mName.data);
    glm::mat4 NodeTransformation(AssimpGLMHelpers::ConvertMatrixToGLMFormat(pNode->mTransformation));
    glm::mat4 GlobalTransformation = ParentTransform * NodeTransformation;
    if (m_BoneNameToIndexMap.find(NodeName) != m_BoneNameToIndexMap.end()) {
        unsigned int BoneIndex = m_BoneNameToIndexMap[NodeName];
        m_BoneInfo[BoneIndex].FinalTransformation = GlobalTransformation * m_BoneInfo[BoneIndex].OffsetMatrix;
    }
    for (unsigned int i = 0 ; i < pNode->mNumChildren ; i++) {
        ReadNodeHierarchy(pNode->mChildren[i], GlobalTransformation);
    }
}

std::string SkinnedModel::GetDirFromFilename(const std::string& Filename)
{
    // Extract the directory part from the file name
    std::string::size_type SlashIndex;
    SlashIndex = Filename.find_last_of("\\");

    if (SlashIndex == -1) {
        SlashIndex = Filename.find_last_of("/");
    }
    std::string Dir;

    if (SlashIndex == std::string::npos) {
        Dir = ".";
    }
    else if (SlashIndex == 0) {
        Dir = "/";
    }
    else {
        Dir = Filename.substr(0, SlashIndex);
    }
    return Dir;
}