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
#include "GLMhelpers.h"
#include <filesystem>

void SkinnedModel::Clear()
{
    if (m_Buffers[0] != 0)
    {
        glDeleteBuffers(sizeof(m_Buffers) / sizeof(m_Buffers[0]), m_Buffers);
    }

    if (m_VAO != 0)
    {
        glDeleteVertexArrays(1, &m_VAO);
        m_VAO = 0;
    }
}

bool SkinnedModel::LoadMesh(const std::string &Filename)
{
    // Release the previously loaded mesh (if it exists)
    Clear();

    // Create the VAO
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    // Create the buffers for the vertices attributes
    glGenBuffers(sizeof(m_Buffers) / sizeof(m_Buffers[0]), m_Buffers);

    bool Ret = false;
    std::filesystem::path p = Filename;
    std::cout << "Loading mesh: " << std::filesystem::absolute(p) << std::endl;
    pScene = Importer.ReadFile(Filename.c_str(), ASSIMP_LOAD_FLAGS);

    if (pScene)
    {
        m_GlobalInverseTransform = GLMHelpers::ConvertMatrixToGLMFormat(pScene->mRootNode->mTransformation);
        m_GlobalInverseTransform = glm::inverse(m_GlobalInverseTransform);
        Ret = InitFromScene(pScene, Filename);
    }
    else
    {
        std::cout << "Error parsing '" << Filename << "': '" << Importer.GetErrorString() << "'" << std::endl;
    }

    // Make sure the VAO is not changed from the outside
    glBindVertexArray(0);
    bone_leap_map =
        {
            {"Wrist", 0},
            {"Elbow", 1},
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
            {"pinky_c", 21}};
    return Ret;
}

bool SkinnedModel::InitFromScene(const aiScene *pScene, const std::string &Filename)
{
    m_Meshes.resize(pScene->mNumMeshes);
    m_Materials.resize(pScene->mNumMaterials);

    unsigned int NumVertices = 0;
    unsigned int NumIndices = 0;

    CountVerticesAndIndices(pScene, NumVertices, NumIndices);

    ReserveSpace(NumVertices, NumIndices);

    InitAllMeshes(pScene);

    if (!InitMaterials(pScene, Filename))
    {
        return false;
    }

    PopulateBuffers();

    return glGetError() == GL_NO_ERROR;
}

void SkinnedModel::CountVerticesAndIndices(const aiScene *pScene, unsigned int &NumVertices, unsigned int &NumIndices)
{
    for (unsigned int i = 0; i < m_Meshes.size(); i++)
    {
        m_Meshes[i].MaterialIndex = pScene->mMeshes[i]->mMaterialIndex;
        m_Meshes[i].NumIndices = pScene->mMeshes[i]->mNumFaces * 3;
        m_Meshes[i].BaseVertex = NumVertices;
        m_Meshes[i].BaseIndex = NumIndices;

        NumVertices += pScene->mMeshes[i]->mNumVertices;
        NumIndices += m_Meshes[i].NumIndices;
    }
}

void SkinnedModel::ReserveSpace(unsigned int NumVertices, unsigned int NumIndices)
{
    m_Positions.reserve(NumVertices);
    m_Normals.reserve(NumVertices);
    m_TexCoords.reserve(NumVertices);
    m_VertColors.reserve(NumVertices);
    m_Indices.reserve(NumIndices);
    m_Bones.resize(NumVertices);
}

void SkinnedModel::InitAllMeshes(const aiScene *pScene)
{
    for (unsigned int i = 0; i < m_Meshes.size(); i++)
    {
        const aiMesh *paiMesh = pScene->mMeshes[i];
        InitSingleMesh(i, paiMesh);
    }
    // assert that all vertices have weights that sums to roughly one
    for (unsigned int i = 0; i < m_Meshes.size(); i++)
    {
        const aiMesh *paiMesh = pScene->mMeshes[i];
        if (paiMesh->mNumBones == 0)
            continue;
        for (unsigned int i = 0; i < m_Bones.size(); i++)
        {
            float sum_weight = m_Bones[i].sum_weights();
            if (abs(sum_weight - 1.0f) > 0.01f)
            {
                std::cout << "vertex " << i << " has sum of weights " << sum_weight << std::endl;
                exit(1);
            }
        }
    }
    // flip map
    for (std::map<std::string, unsigned int>::iterator i = m_BoneNameToIndexMap.begin(); i != m_BoneNameToIndexMap.end(); ++i)
        m_BoneIndexToNameMap[i->second] = i->first;
    glm::mat4 iden = glm::mat4(1.0f);
    ReadNodeHierarchy(pScene->mRootNode, iden);
}

void SkinnedModel::InitSingleMesh(unsigned int MeshIndex, const aiMesh *paiMesh)
{
    const aiVector3D Zero3D(0.0f, 0.0f, 0.0f);
    const aiColor4D Zero4D(0.0f, 0.0f, 0.0f, 1.0f);

    // Populate the vertex attribute vectors
    for (unsigned int i = 0; i < paiMesh->mNumVertices; i++)
    {

        const aiVector3D &pPos = paiMesh->mVertices[i];
        m_Positions.push_back(glm::vec3(pPos.x, pPos.y, pPos.z));

        if (paiMesh->mNormals)
        {
            const aiVector3D &pNormal = paiMesh->mNormals[i];
            m_Normals.push_back(glm::vec3(pNormal.x, pNormal.y, pNormal.z));
        }
        else
        {
            aiVector3D Normal(0.0f, 1.0f, 0.0f);
            m_Normals.push_back(glm::vec3(Normal.x, Normal.y, Normal.z));
        }

        const aiVector3D &pTexCoord = paiMesh->HasTextureCoords(0) ? paiMesh->mTextureCoords[0][i] : Zero3D;
        m_TexCoords.push_back(glm::vec2(pTexCoord.x, pTexCoord.y));

        const aiColor4D &pVertColor = paiMesh->HasVertexColors(0) ? paiMesh->mColors[0][i] : Zero4D;
        m_VertColors.push_back(glm::vec3(pVertColor.r, pVertColor.g, pVertColor.b));
    }

    LoadMeshBones(MeshIndex, paiMesh);

    // Populate the index buffer
    for (unsigned int i = 0; i < paiMesh->mNumFaces; i++)
    {
        const aiFace &Face = paiMesh->mFaces[i];
        //        printf("num indices %d\n", Face.mNumIndices);
        //        assert(Face.mNumIndices == 3);
        m_Indices.push_back(Face.mIndices[0]);
        if (m_leftChirality)
        {
            m_Indices.push_back(Face.mIndices[1]);
            m_Indices.push_back(Face.mIndices[2]);
        }
        else
        {
            m_Indices.push_back(Face.mIndices[2]);
            m_Indices.push_back(Face.mIndices[1]);
        }
    }
}

void SkinnedModel::LoadMeshBones(unsigned int MeshIndex, const aiMesh *pMesh)
{
    if (pMesh->mNumBones == 0)
        return;
    for (unsigned int i = 0; i < pMesh->mNumBones; i++)
    {
        LoadSingleBone(MeshIndex, pMesh->mBones[i]);
    }
}

std::string SkinnedModel::getBoneName(unsigned int index)
{
    if (index >= m_BoneIndexToNameMap.size())
        return "";
    return m_BoneIndexToNameMap[index];
}

void SkinnedModel::LoadSingleBone(unsigned int MeshIndex, const aiBone *pBone)
{
    int BoneId = GetBoneId(pBone);

    if (BoneId == m_BoneInfo.size())
    {
        BoneInfo bi(GLMHelpers::ConvertMatrixToGLMFormat(pBone->mOffsetMatrix));
        m_BoneInfo.push_back(bi);
    }

    for (unsigned int i = 0; i < pBone->mNumWeights; i++)
    {
        const aiVertexWeight &vw = pBone->mWeights[i];
        unsigned int GlobalVertexID = m_Meshes[MeshIndex].BaseVertex + pBone->mWeights[i].mVertexId;
        m_Bones[GlobalVertexID].AddBoneData(BoneId, vw.mWeight);
    }
}

int SkinnedModel::GetBoneId(const aiBone *pBone)
{
    int BoneIndex = 0;
    std::string BoneName(pBone->mName.C_Str());

    if (m_BoneNameToIndexMap.find(BoneName) == m_BoneNameToIndexMap.end())
    {
        // Allocate an index for a new bone
        BoneIndex = (int)m_BoneNameToIndexMap.size();
        m_BoneNameToIndexMap[BoneName] = BoneIndex;
    }
    else
    {
        BoneIndex = m_BoneNameToIndexMap[BoneName];
    }

    return BoneIndex;
}

bool SkinnedModel::InitMaterials(const aiScene *pScene, const std::string &Filename)
{
    std::string Dir = GetDirFromFilename(Filename);

    bool Ret = true;

    printf("Num materials: %d\n", pScene->mNumMaterials);

    // Initialize the materials
    for (unsigned int i = 0; i < pScene->mNumMaterials; i++)
    {
        const aiMaterial *pMaterial = pScene->mMaterials[i];

        LoadTextures(Dir, pMaterial, i);

        LoadColors(pMaterial, i);
    }

    return Ret;
}

void SkinnedModel::LoadTextures(const std::string &Dir, const aiMaterial *pMaterial, int index)
{
    LoadDiffuseTexture(Dir, pMaterial, index);
    LoadSpecularTexture(Dir, pMaterial, index);
}

void SkinnedModel::LoadDiffuseTexture(const std::string &Dir, const aiMaterial *pMaterial, int index)
{
    m_Materials[index].pDiffuse = NULL;
    if (m_externalTextureFileName != "") // bypass model texture
    {
        std::filesystem::path p = m_externalTextureFileName;
        std::cout << "Loading diffuse texture: " << std::filesystem::absolute(p) << std::endl;
        m_Materials[index].pDiffuse = new Texture(m_externalTextureFileName.c_str(), GL_TEXTURE_2D);
        if (!m_Materials[index].pDiffuse->init_from_file(GL_LINEAR, GL_REPEAT))
        {
            std::cout << "Error loading diffuse texture." << std::endl;
            exit(1);
        }
    }
    else
    {
        if (pMaterial->GetTextureCount(aiTextureType_DIFFUSE) > 0)
        {
            aiString Path;

            if (pMaterial->GetTexture(aiTextureType_DIFFUSE, 0, &Path, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS)
            {
                std::string p(Path.data);

                if (p.substr(0, 2) == ".\\")
                {
                    p = p.substr(2, p.size() - 2);
                }

                std::string FullPath = Dir + "/" + p;

                m_Materials[index].pDiffuse = new Texture(FullPath.c_str(), GL_TEXTURE_2D);

                if (!m_Materials[index].pDiffuse->init_from_file(GL_LINEAR, GL_REPEAT))
                {
                    std::cout << "Error loading diffuse texture '" << FullPath << "'" << std::endl;
                }
                else
                {
                    std::cout << "Loaded diffuse texture '" << FullPath << "'" << std::endl;
                }
            }
        }
    }
}

void SkinnedModel::LoadSpecularTexture(const std::string &Dir, const aiMaterial *pMaterial, int index)
{
    m_Materials[index].pSpecularExponent = NULL;

    if (pMaterial->GetTextureCount(aiTextureType_SHININESS) > 0)
    {
        aiString Path;

        if (pMaterial->GetTexture(aiTextureType_SHININESS, 0, &Path, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS)
        {
            std::string p(Path.data);

            if (p == "C:\\\\")
            {
                p = "";
            }
            else if (p.substr(0, 2) == ".\\")
            {
                p = p.substr(2, p.size() - 2);
            }

            std::string FullPath = Dir + "/" + p;

            m_Materials[index].pSpecularExponent = new Texture(FullPath.c_str(), GL_TEXTURE_2D);

            if (!m_Materials[index].pSpecularExponent->init_from_file())
            {
                printf("Error loading specular texture '%s'\n", FullPath.c_str());
                exit(0);
            }
            else
            {
                printf("Loaded specular texture '%s'\n", FullPath.c_str());
            }
        }
    }
}

void SkinnedModel::LoadColors(const aiMaterial *pMaterial, int index)
{
    aiColor3D AmbientColor(0.0f, 0.0f, 0.0f);
    glm::vec3 AllOnes(1.0f, 1.0f, 1.0f);

    int ShadingModel = 0;
    if (pMaterial->Get(AI_MATKEY_SHADING_MODEL, ShadingModel) == AI_SUCCESS)
    {
        std::cout << "Assimp shading model: " << ShadingModel << std::endl;
    }

    if (pMaterial->Get(AI_MATKEY_COLOR_AMBIENT, AmbientColor) == AI_SUCCESS)
    {
        std::cout << "Loaded ambient color: [" << AmbientColor.r << " " << AmbientColor.g << " " << AmbientColor.b << "]" << std::endl;
        m_Materials[index].AmbientColor.r = AmbientColor.r;
        m_Materials[index].AmbientColor.g = AmbientColor.g;
        m_Materials[index].AmbientColor.b = AmbientColor.b;
    }
    else
    {
        m_Materials[index].AmbientColor = AllOnes;
    }

    aiColor3D DiffuseColor(0.0f, 0.0f, 0.0f);

    if (pMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, DiffuseColor) == AI_SUCCESS)
    {
        std::cout << "Loaded diffuse color: [" << DiffuseColor.r << " " << DiffuseColor.g << " " << DiffuseColor.b << "]" << std::endl;
        m_Materials[index].DiffuseColor.r = DiffuseColor.r;
        m_Materials[index].DiffuseColor.g = DiffuseColor.g;
        m_Materials[index].DiffuseColor.b = DiffuseColor.b;
    }

    aiColor3D SpecularColor(0.0f, 0.0f, 0.0f);

    if (pMaterial->Get(AI_MATKEY_COLOR_SPECULAR, SpecularColor) == AI_SUCCESS)
    {
        std::cout << "Loaded specular color: [" << SpecularColor.r << " " << SpecularColor.g << " " << SpecularColor.b << "]" << std::endl;
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

    glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[VERTEX_COLOR_VB]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(m_VertColors[0]) * m_VertColors.size(), &m_VertColors[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(VERTEX_COLOR_LOCATION);
    glVertexAttribPointer(VERTEX_COLOR_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);

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
    glVertexAttribIPointer(BONE_ID_LOCATION0, MAX_NUM_BONES_PER_VERTEX - 2, GL_INT, sizeof(VertexBoneData), (const GLvoid *)0);

    glEnableVertexAttribArray(BONE_ID_LOCATION1);
    glVertexAttribIPointer(BONE_ID_LOCATION1, MAX_NUM_BONES_PER_VERTEX - 4, GL_INT, sizeof(VertexBoneData), (const GLvoid *)((MAX_NUM_BONES_PER_VERTEX - 2) * sizeof(unsigned int)));

    glEnableVertexAttribArray(BONE_WEIGHT_LOCATION0);
    glVertexAttribPointer(BONE_WEIGHT_LOCATION0, MAX_NUM_BONES_PER_VERTEX - 2, GL_FLOAT, GL_FALSE, sizeof(VertexBoneData),
                          (const GLvoid *)(MAX_NUM_BONES_PER_VERTEX * sizeof(unsigned int)));

    glEnableVertexAttribArray(BONE_WEIGHT_LOCATION1);
    size_t offset = ((MAX_NUM_BONES_PER_VERTEX) * sizeof(unsigned int)) + ((MAX_NUM_BONES_PER_VERTEX - 2) * sizeof(float));
    glVertexAttribPointer(BONE_WEIGHT_LOCATION1, MAX_NUM_BONES_PER_VERTEX - 4, GL_FLOAT, GL_FALSE, sizeof(VertexBoneData),
                          (const GLvoid *)offset);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Buffers[INDEX_BUFFER]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_Indices[0]) * m_Indices.size(), &m_Indices[0], GL_STATIC_DRAW);
    // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    /* camera texture */
    // glGenTextures(1, &m_cam_texture);
    // glBindTexture(GL_TEXTURE_2D, m_cam_texture);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // GL_CLAMP_TO_EDGE
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT); // GL_CLAMP_TO_EDGE
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA /*GL_RGBA16F*/, m_camWidth, m_camHeight, 0, GL_BGRA /* GL_RGBA*/,
    //              GL_UNSIGNED_BYTE, NULL);
    /* camera texture */
}

void SkinnedModel::Render(Shader &shader, unsigned int camTex, bool useFBO)
{
    shader.use();
    shader.setInt("src", 0);
    shader.setInt("projTexture", 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, camTex);
    if (useFBO)
    {
        m_fbo.bind();
        glEnable(GL_DEPTH_TEST);
    }
    glBindVertexArray(m_VAO);

    for (unsigned int i = 0; i < m_Meshes.size(); i++)
    {
        unsigned int MaterialIndex = m_Meshes[i].MaterialIndex;

        assert(MaterialIndex < m_Materials.size());

        if (m_Materials[MaterialIndex].pDiffuse)
        {
            m_Materials[MaterialIndex].pDiffuse->bind(GL_TEXTURE0);
        }

        if (m_Materials[MaterialIndex].pSpecularExponent)
        {
            m_Materials[MaterialIndex].pSpecularExponent->bind(GL_TEXTURE6);
        }

        glDrawElementsBaseVertex(GL_TRIANGLES,
                                 m_Meshes[i].NumIndices,
                                 GL_UNSIGNED_INT,
                                 (void *)(sizeof(unsigned int) * m_Meshes[i].BaseIndex),
                                 m_Meshes[i].BaseVertex);
    }

    // Make sure the VAO is not changed from the outside
    glBindVertexArray(0);
    if (useFBO)
    {
        m_fbo.unbind();
        glDisable(GL_DEPTH_TEST);
    }
}
void SkinnedModel::Render(SkinningShader &shader, const std::vector<glm::mat4> &bones_to_world,
                          const glm::mat4 &local_to_world, const bool use_bones, Texture *customDiffuseTexture)
{
    shader.use();
    shader.SetMaterial(GetMaterial());
    std::vector<glm::mat4> transforms;
    // todo: for unknown reason passing local_to_world fails, probably memory leak.
    GetBoneTransforms(transforms, bones_to_world, local_to_world, use_bones);
    // GetBoneTransformsHack(transforms, bones_to_world);
    for (unsigned int i = 0; i < transforms.size(); i++)
    {
        shader.SetBoneTransform(i, transforms[i]);
    }
    glBindVertexArray(m_VAO);

    for (unsigned int i = 0; i < m_Meshes.size(); i++)
    {
        unsigned int MaterialIndex = m_Meshes[i].MaterialIndex;

        assert(MaterialIndex < m_Materials.size());

        if (customDiffuseTexture != NULL)
        {
            customDiffuseTexture->bind(GL_TEXTURE0);
        }
        else
        {
            if (m_Materials[MaterialIndex].pDiffuse)
            {
                m_Materials[MaterialIndex].pDiffuse->bind(GL_TEXTURE0);
            }
        }
        if (m_Materials[MaterialIndex].pSpecularExponent)
        {
            m_Materials[MaterialIndex].pSpecularExponent->bind(GL_TEXTURE6);
        }

        glDrawElementsBaseVertex(GL_TRIANGLES,
                                 m_Meshes[i].NumIndices,
                                 GL_UNSIGNED_INT,
                                 (void *)(sizeof(unsigned int) * m_Meshes[i].BaseIndex),
                                 m_Meshes[i].BaseVertex);
    }
    // Make sure the VAO is not changed from the outside
    glBindVertexArray(0);
}

void SkinnedModel::Render(SkinningShader &shader, const std::vector<glm::mat4> &bones_to_world,
                          const glm::mat4 &local_to_world, unsigned int camTex, bool useFBO, bool use_bones)
{
    shader.use();
    shader.SetMaterial(this->GetMaterial());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, camTex);
    if (useFBO)
    {
        m_fbo.bind();
        glEnable(GL_DEPTH_TEST);
    }
    // glActiveTexture(GL_TEXTURE1);
    // glBindTexture(GL_TEXTURE_2D, m_cam_texture);
    std::vector<glm::mat4> Transforms;
    this->GetBoneTransforms(Transforms, bones_to_world, local_to_world, use_bones);
    for (unsigned int i = 0; i < Transforms.size(); i++)
    {
        shader.SetBoneTransform(i, Transforms[i]);
    }
    glBindVertexArray(m_VAO);

    for (unsigned int i = 0; i < m_Meshes.size(); i++)
    {
        unsigned int MaterialIndex = m_Meshes[i].MaterialIndex;

        assert(MaterialIndex < m_Materials.size());

        if (m_Materials[MaterialIndex].pDiffuse)
        {
            m_Materials[MaterialIndex].pDiffuse->bind(GL_TEXTURE0);
        }

        if (m_Materials[MaterialIndex].pSpecularExponent)
        {
            m_Materials[MaterialIndex].pSpecularExponent->bind(GL_TEXTURE6);
        }

        glDrawElementsBaseVertex(GL_TRIANGLES,
                                 m_Meshes[i].NumIndices,
                                 GL_UNSIGNED_INT,
                                 (void *)(sizeof(unsigned int) * m_Meshes[i].BaseIndex),
                                 m_Meshes[i].BaseVertex);
    }

    // Make sure the VAO is not changed from the outside
    glBindVertexArray(0);
    if (useFBO)
    {
        m_fbo.unbind();
        glDisable(GL_DEPTH_TEST);
    }
}

const Material &SkinnedModel::GetMaterial()
{
    for (unsigned int i = 0; i < m_Materials.size(); i++)
    {
        if (m_Materials[i].AmbientColor != glm::vec3(0.0f, 0.0f, 0.0f))
        {
            return m_Materials[i];
        }
    }

    return m_Materials[0];
}

glm::vec3 SkinnedModel::getCenterOfMass()
{
    glm::vec3 center_of_mass = glm::vec3(0.0f, 0.0f, 0.0f);
    for (unsigned int i = 0; i < m_Positions.size(); i++)
    {
        center_of_mass += m_Positions[i];
    }
    center_of_mass /= (float)m_Positions.size();
    return center_of_mass;
};

void SkinnedModel::GetLocalToBoneTransforms(std::vector<glm::mat4> &transforms, bool inverse, bool only_leap_bones)
{
    if (only_leap_bones)
    {
        transforms.resize(bone_leap_map.size());
        int i = 0;
        for (auto const &x : bone_leap_map)
        {
            unsigned int bone_index = m_BoneNameToIndexMap[x.first];
            if (inverse)
                transforms[i] = glm::inverse(m_BoneInfo[bone_index].LocalToBoneTransform);
            else
                transforms[i] = m_BoneInfo[bone_index].LocalToBoneTransform;
            i++;
        }
    }
    else
    {
        transforms.resize(m_BoneInfo.size());
        for (unsigned int i = 0; i < m_BoneInfo.size(); i++)
        {
            if (inverse)
                transforms[i] = glm::inverse(m_BoneInfo[i].LocalToBoneTransform);
            else
                transforms[i] = m_BoneInfo[i].LocalToBoneTransform;
        }
    }
}

void SkinnedModel::GetBoneFinalTransforms(std::vector<glm::mat4> &transforms)
{
    transforms.resize(m_BoneInfo.size());
    for (unsigned int i = 0; i < m_BoneInfo.size(); i++)
    {
        transforms[i] = m_BoneInfo[i].FinalTransformation;
    }
}

void SkinnedModel::GetBoneTransformRelativeToParent(std::vector<glm::mat4> &transforms)
{
    transforms.resize(m_BoneInfo.size());
    aiNode *pNode = pScene->mRootNode;
    std::string NodeName(pNode->mName.data);
    glm::mat4 NodeTransformation(GLMHelpers::ConvertMatrixToGLMFormat(pNode->mTransformation));
    // glm::mat4 GlobalTransformation = ParentTransform * NodeTransformation;
    if (m_BoneNameToIndexMap.find(NodeName) != m_BoneNameToIndexMap.end())
    {
        unsigned int BoneIndex = m_BoneNameToIndexMap[NodeName];
        transforms[BoneIndex] = NodeTransformation;
    }
}

void SkinnedModel::GetBoneTransforms(std::vector<glm::mat4> &transforms, const std::vector<glm::mat4> &bones_to_world, const glm::mat4 &local_to_world, const bool use_bones)
{
    transforms.resize(m_BoneInfo.size());
    if (use_bones)
    {
        // default bind pose using bones
        // "FinalTransfomation" was obtained with read node hierarchy
        for (unsigned int i = 0; i < m_BoneInfo.size(); i++)
        {
            transforms[i] = local_to_world * m_BoneInfo[i].FinalTransformation;
        }
    }
    else
    {
        // default bind pose not using bones
        glm::mat4 iden = glm::mat4(1.0f);
        for (unsigned int i = 0; i < m_BoneInfo.size(); i++)
        {
            transforms[i] = local_to_world * iden;
        }
    }
    // skin the mesh using the bone to world transforms from leap
    // offset matrix is local to bone matrix ("inverse bind pose")
    if (bones_to_world.size() > 0)
    {
        for (auto const &x : bone_leap_map)
        {
            unsigned int bone_index = m_BoneNameToIndexMap[x.first];
            transforms[bone_index] = bones_to_world[x.second] * m_BoneInfo[bone_index].LocalToBoneTransform;
        }
    }
}

void SkinnedModel::GetBoneTransformsHack(std::vector<glm::mat4> &transforms, const std::vector<glm::mat4> bones_to_world)
{
    transforms.resize(m_BoneInfo.size());
    // skin the mesh using the bone to world transforms from leap
    // offset matrix is local to bone matrix ("inverse bind pose")
    if (bones_to_world.size() > 0)
    {
        for (auto const &x : bone_leap_map)
        {
            unsigned int bone_index = m_BoneNameToIndexMap[x.first];
            transforms[bone_index] = bones_to_world[x.second] * m_BoneInfo[bone_index].LocalToBoneTransform;
        }
    }
}

void SkinnedModel::ReadNodeHierarchy(const aiNode *pNode, const glm::mat4 &ParentTransform)
{
    // if the mesh is in bind pose, this is expected to set all FinalTransformation to identity
    // because mTransformation is child to parent transform (e.g. bone to local for root), and local to bone cancles it out.
    std::string NodeName(pNode->mName.data);
    glm::mat4 NodeTransformation(GLMHelpers::ConvertMatrixToGLMFormat(pNode->mTransformation));
    glm::mat4 GlobalTransformation = ParentTransform * NodeTransformation;
    if (m_BoneNameToIndexMap.find(NodeName) != m_BoneNameToIndexMap.end())
    {
        unsigned int BoneIndex = m_BoneNameToIndexMap[NodeName];
        m_BoneInfo[BoneIndex].FinalTransformation = GlobalTransformation * m_BoneInfo[BoneIndex].LocalToBoneTransform;
    }
    for (unsigned int i = 0; i < pNode->mNumChildren; i++)
    {
        ReadNodeHierarchy(pNode->mChildren[i], GlobalTransformation);
    }
}

std::string SkinnedModel::GetDirFromFilename(const std::string &Filename)
{
    // Extract the directory part from the file name
    std::string::size_type SlashIndex;
    SlashIndex = Filename.find_last_of("\\");

    if (SlashIndex == -1)
    {
        SlashIndex = Filename.find_last_of("/");
    }
    std::string Dir;

    if (SlashIndex == std::string::npos)
    {
        Dir = ".";
    }
    else if (SlashIndex == 0)
    {
        Dir = "/";
    }
    else
    {
        Dir = Filename.substr(0, SlashIndex);
    }
    return Dir;
}