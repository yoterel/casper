#include "grid.h"
#include "helpers.h"

Grid::Grid(const GLuint xPointCount, const GLuint yPointCount, const GLfloat xSpacing, const GLfloat ySpacing) : Grid_VBO(0),
                                                                                                                 Grid_VAO(0),
                                                                                                                 Grid_EBO(0)
{
    m_xPointCount = xPointCount;
    m_yPointCount = yPointCount;
    m_xSpacing = xSpacing;
    m_ySpacing = ySpacing;
    constructGrid();
    M = AssembleM();
}

Grid::~Grid()
{
    glDeleteVertexArrays(1, &Grid_VAO);
    glDeleteBuffers(1, &Grid_VBO);
    glDeleteBuffers(1, &Grid_EBO);
}

GLfloat *Grid::ComputePointCoordinates(GLuint pointIndex)
{
    static GLfloat pt[3];

    GLfloat width = m_xSpacing * (m_xPointCount - 1);         // 0.2*10 = 2.0
    GLfloat height = m_ySpacing * (m_yPointCount - 1);        // 0.2*10 = 2.0
    GLfloat minX = -width / 2;                                // -2.0/2 = -1.0
    GLfloat minY = -height / 2;                               // -2.0/2 = -1.0
    pt[0] = minX + m_xSpacing * (pointIndex % m_xPointCount); //-1.0+0.2*(34%11) = x
    pt[1] = minY + m_ySpacing * (pointIndex / m_xPointCount); //-1.0+0.2*(34/11) = y
    pt[2] = 0;
    return pt;
}

void Grid::ComputePointCoordinate(int pointIndex,
                                  float pt[3])
{
    float *tmp = ComputePointCoordinates(pointIndex);
    pt[0] = tmp[0];
    pt[1] = tmp[1];
    pt[2] = tmp[2];
}

// assemble grid points for fruther calculation
cv::Mat Grid::AssembleM()
{
    // change the size of the grid
    cv::Mat a = cv::Mat::zeros(2, m_xPointCount * m_yPointCount, CV_32F); // 2 x 1681 x float
    double width = m_xSpacing * (m_xPointCount - 1);                      // 2.0
    double height = m_ySpacing * (m_yPointCount - 1);                     // 2.0
    double minX = -width / 2;
    double minY = -height / 2;

    for (int i = 0; i < m_xPointCount; i++)
    {
        for (int j = 0; j < m_yPointCount; j++)
        {
            double x = minX + i * m_xSpacing;
            double y = minY + j * m_ySpacing;

            // save the mat points to get v
            a.at<float>(0, i * m_yPointCount + j) = static_cast<float>(x);
            a.at<float>(1, i * m_yPointCount + j) = static_cast<float>(y);
        }
    }

    return a;
}

void Grid::initGLBuffers()
{
    glGenVertexArrays(1, &Grid_VAO);
    glGenBuffers(1, &Grid_VBO);
    glGenBuffers(1, &Grid_EBO);

    glBindVertexArray(Grid_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, Grid_VBO);
    glBufferData(GL_ARRAY_BUFFER, Grid_vertices.size() * sizeof(glm::vec3), &Grid_vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Grid_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, Grid_indices.size() * sizeof(glm::ivec3), &Grid_indices[0], GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(glm::vec3), static_cast<void *>(nullptr));
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(glm::vec3), reinterpret_cast<void *>(1 * sizeof(glm::vec3)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(glm::vec3), reinterpret_cast<void *>(2 * sizeof(glm::vec3)));
    glEnableVertexAttribArray(2);
}

void Grid::updateGLBuffers()
{
    glBindBuffer(GL_ARRAY_BUFFER, Grid_VBO);
    glBufferData(GL_ARRAY_BUFFER, Grid_vertices.size() * sizeof(glm::vec3), &Grid_vertices[0], GL_DYNAMIC_DRAW);
    // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Grid_EBO); // todo: these don't really need to be uploaded every frame though
    // glBufferData(GL_ELEMENT_ARRAY_BUFFER, Grid_indices.size() * sizeof(glm::ivec3), &Grid_indices[0], GL_STATIC_DRAW);
}

void Grid::constructGrid()
{
    float width = (m_xPointCount - 1) * m_xSpacing;
    float height = (m_yPointCount - 1) * m_ySpacing;
    float minX = -width / 2;
    float minY = -height / 2;

    Grid_vertices.clear();
    Grid_indices.clear();

    GLuint nrQuads = (m_xPointCount - 1) * (m_yPointCount - 1); // 1600
    for (GLuint i = 0; i < nrQuads; i++)
    {
        GLuint k = i + i / (m_xPointCount - 1);
        GLuint a = k;                     // LU
        GLuint b = k + 1;                 // RU
        GLuint c = k + 1 + m_xPointCount; // RD
        GLuint d = k + m_xPointCount;     // LD
        GLfloat aPt[3], bPt[3], cPt[3], dPt[3];
        ComputePointCoordinate(a, aPt);
        ComputePointCoordinate(b, bPt);
        ComputePointCoordinate(c, cPt);
        ComputePointCoordinate(d, dPt);

        // Triangle 1 c d a
        // Pos Color Coord
        Grid_vertices.push_back(glm::vec3(cPt[0], cPt[1], cPt[2]));
        Grid_vertices.push_back(glm::vec3(0.0, 1.0, 0.0));
        Grid_vertices.push_back(glm::vec3((cPt[0] - minX) / width, (cPt[1] - minY) / height, 0.0));

        Grid_vertices.push_back(glm::vec3(dPt[0], dPt[1], dPt[2]));
        Grid_vertices.push_back(glm::vec3(0.0, 0.0, 1.0));
        Grid_vertices.push_back(glm::vec3((dPt[0] - minX) / width, (dPt[1] - minY) / height, 0.0));

        Grid_vertices.push_back(glm::vec3(aPt[0], aPt[1], aPt[2]));
        Grid_vertices.push_back(glm::vec3(1.0, 0.0, 0.0));
        Grid_vertices.push_back(glm::vec3((aPt[0] - minX) / width, (aPt[1] - minY) / height, 0.0));

        Grid_indices.push_back(glm::ivec3(Grid_vertices.size() / 3 - 3, Grid_vertices.size() / 3 - 2, Grid_vertices.size() / 3 - 1));

        // Triangle 2 a b c
        // Pos Color Coord
        Grid_vertices.push_back(glm::vec3(aPt[0], aPt[1], aPt[2]));
        Grid_vertices.push_back(glm::vec3(1.0, 0.0, 0.0));
        Grid_vertices.push_back(glm::vec3((aPt[0] - minX) / width, (aPt[1] - minY) / height, 0.0));

        Grid_vertices.push_back(glm::vec3(bPt[0], bPt[1], bPt[2]));
        Grid_vertices.push_back(glm::vec3(0.0, 1.0, 0.0));
        Grid_vertices.push_back(glm::vec3((bPt[0] - minX) / width, (bPt[1] - minY) / height, 0.0));

        Grid_vertices.push_back(glm::vec3(cPt[0], cPt[1], cPt[2]));
        Grid_vertices.push_back(glm::vec3(0.0, 0.0, 1.0));
        Grid_vertices.push_back(glm::vec3((cPt[0] - minX) / width, (cPt[1] - minY) / height, 0.0));

        Grid_indices.push_back(glm::ivec3(Grid_vertices.size() / 3 - 3, Grid_vertices.size() / 3 - 2, Grid_vertices.size() / 3 - 1));
    }

    // printf("vertices length:%d indices length:%d\n", static_cast<int>(Grid_vertices.size()), static_cast<int>(Grid_indices.size()));
}

void Grid::constructDeformedGrid(cv::Mat fv)
{
    double width = (m_xPointCount - 1) * m_xSpacing;
    double height = (m_yPointCount - 1) * m_ySpacing;
    double minX = -width / 2;
    double minY = -height / 2;

    Grid_vertices.clear();
    Grid_indices.clear();

    int nrQuads = (m_xPointCount - 1) * (m_yPointCount - 1);
    for (int i = 0; i < nrQuads; i++)
    {
        int k = i + i / (m_xPointCount - 1);
        int a = k;
        int b = k + 1;
        int c = k + 1 + m_xPointCount;
        int d = k + m_xPointCount;
        double aPt[3], bPt[3], cPt[3], dPt[3];
        float aPtIm[3], bPtIm[3], cPtIm[3], dPtIm[3];

        ComputePointCoordinate(a, aPtIm);
        ComputePointCoordinate(b, bPtIm);
        ComputePointCoordinate(c, cPtIm);
        ComputePointCoordinate(d, dPtIm);

        // get the deformed points
        aPt[0] = fv.at<float>(0, a);
        aPt[1] = fv.at<float>(1, a);
        aPt[2] = 0;

        bPt[0] = fv.at<float>(0, b);
        bPt[1] = fv.at<float>(1, b);
        bPt[2] = 0;

        cPt[0] = fv.at<float>(0, c);
        cPt[1] = fv.at<float>(1, c);
        cPt[2] = 0;

        dPt[0] = fv.at<float>(0, d);
        dPt[1] = fv.at<float>(1, d);
        dPt[2] = 0;

        // mapping the actual points
        //  Triangle 1 c d a
        //  Pos Color Coord
        Grid_vertices.push_back(glm::vec3(cPt[0], cPt[1], cPt[2]));
        Grid_vertices.push_back(glm::vec3(0.0, 1.0, 0.0));
        Grid_vertices.push_back(glm::vec3((cPtIm[1] - minX) / width, (cPtIm[0] - minY) / height, 0.0));

        Grid_vertices.push_back(glm::vec3(dPt[0], dPt[1], dPt[2]));
        Grid_vertices.push_back(glm::vec3(0.0, 0.0, 1.0));
        Grid_vertices.push_back(glm::vec3((dPtIm[1] - minX) / width, (dPtIm[0] - minY) / height, 0.0));

        Grid_vertices.push_back(glm::vec3(aPt[0], aPt[1], aPt[2]));
        Grid_vertices.push_back(glm::vec3(1.0, 1.0, 0.0));
        Grid_vertices.push_back(glm::vec3((aPtIm[1] - minX) / width, (aPtIm[0] - minY) / height, 0.0));

        Grid_indices.push_back(glm::ivec3(Grid_vertices.size() / 3 - 3, Grid_vertices.size() / 3 - 2, Grid_vertices.size() / 3 - 1));

        // Triangle 2 a b c
        // Pos Color Coord
        Grid_vertices.push_back(glm::vec3(aPt[0], aPt[1], aPt[2]));
        Grid_vertices.push_back(glm::vec3(1.0, 1.0, 0.0));
        Grid_vertices.push_back(glm::vec3((aPtIm[1] - minX) / width, (aPtIm[0] - minY) / height, 0.0));

        Grid_vertices.push_back(glm::vec3(bPt[0], bPt[1], bPt[2]));
        Grid_vertices.push_back(glm::vec3(0.0, 1.0, 0.0));
        Grid_vertices.push_back(glm::vec3((bPtIm[1] - minX) / width, (bPtIm[0] - minY) / height, 0.0));

        Grid_vertices.push_back(glm::vec3(cPt[0], cPt[1], cPt[2]));
        Grid_vertices.push_back(glm::vec3(0.0, 0.0, 1.0));
        Grid_vertices.push_back(glm::vec3((cPtIm[1] - minX) / width, (cPtIm[0] - minY) / height, 0.0));

        Grid_indices.push_back(glm::ivec3(Grid_vertices.size() / 3 - 3, Grid_vertices.size() / 3 - 2, Grid_vertices.size() / 3 - 1));
    }

    // printf("vertices length:%d indices length:%d\n", static_cast<int>(Grid_vertices.size()), static_cast<int>(Grid_indices.size()));
}

void Grid::constructDeformedGridSmooth(cv::Mat fv, int smooth_window)
{
    double width = (m_xPointCount - 1) * m_xSpacing;
    double height = (m_yPointCount - 1) * m_ySpacing;
    double minX = -width / 2;
    double minY = -height / 2;

    Grid_vertices.clear();
    Grid_indices.clear();

    int nrQuads = (m_xPointCount - 1) * (m_yPointCount - 1);
    std::vector<glm::vec3> flat_abcd;
    std::vector<glm::vec3> flat_abcd_texcords;
    for (int i = 0; i < nrQuads; i++)
    {
        int k = i + i / (m_xPointCount - 1);
        int a = k;
        int b = k + 1;
        int c = k + 1 + m_xPointCount;
        int d = k + m_xPointCount;
        double aPt[3], bPt[3], cPt[3], dPt[3];
        float aPtIm[3], bPtIm[3], cPtIm[3], dPtIm[3];

        ComputePointCoordinate(a, aPtIm);
        ComputePointCoordinate(b, bPtIm);
        ComputePointCoordinate(c, cPtIm);
        ComputePointCoordinate(d, dPtIm);

        // get the deformed points
        flat_abcd.push_back(glm::vec3(fv.at<float>(0, a), fv.at<float>(1, a), 0));
        flat_abcd.push_back(glm::vec3(fv.at<float>(0, b), fv.at<float>(1, b), 0));
        flat_abcd.push_back(glm::vec3(fv.at<float>(0, c), fv.at<float>(1, c), 0));
        flat_abcd.push_back(glm::vec3(fv.at<float>(0, d), fv.at<float>(1, d), 0));
        flat_abcd_texcords.push_back(glm::vec3((aPtIm[1] - minX) / width, (aPtIm[0] - minY) / height, 0.0));
        flat_abcd_texcords.push_back(glm::vec3((bPtIm[1] - minX) / width, (bPtIm[0] - minY) / height, 0.0));
        flat_abcd_texcords.push_back(glm::vec3((cPtIm[1] - minX) / width, (cPtIm[0] - minY) / height, 0.0));
        flat_abcd_texcords.push_back(glm::vec3((dPtIm[1] - minX) / width, (dPtIm[0] - minY) / height, 0.0));
    }

    flat_abcd_vec.push_back(flat_abcd);
    std::vector<glm::vec3> abcd_mean = Helpers::accumulate(flat_abcd_vec);
    int diff = flat_abcd_vec.size() - smooth_window;
    if (diff > 0)
    {
        for (int i = 0; i < diff; i++)
            flat_abcd_vec.erase(flat_abcd_vec.begin());
    }

    for (int i = 0; i < abcd_mean.size(); i += 4)
    {
        Grid_vertices.push_back(abcd_mean[i + 2]);
        Grid_vertices.push_back(glm::vec3(0.0, 1.0, 0.0));
        Grid_vertices.push_back(flat_abcd_texcords[i + 2]);

        Grid_vertices.push_back(abcd_mean[i + 3]);
        Grid_vertices.push_back(glm::vec3(0.0, 0.0, 1.0));
        Grid_vertices.push_back(flat_abcd_texcords[i + 3]);

        Grid_vertices.push_back(abcd_mean[i]);
        Grid_vertices.push_back(glm::vec3(1.0, 1.0, 0.0));
        Grid_vertices.push_back(flat_abcd_texcords[i]);

        Grid_indices.push_back(glm::ivec3(Grid_vertices.size() / 3 - 3, Grid_vertices.size() / 3 - 2, Grid_vertices.size() / 3 - 1));

        // Triangle 2 a b c
        // Pos Color Coord
        Grid_vertices.push_back(abcd_mean[i]);
        Grid_vertices.push_back(glm::vec3(1.0, 1.0, 0.0));
        Grid_vertices.push_back(flat_abcd_texcords[i]);

        Grid_vertices.push_back(abcd_mean[i + 1]);
        Grid_vertices.push_back(glm::vec3(0.0, 1.0, 0.0));
        Grid_vertices.push_back(flat_abcd_texcords[i + 1]);

        Grid_vertices.push_back(abcd_mean[i + 2]);
        Grid_vertices.push_back(glm::vec3(0.0, 0.0, 1.0));
        Grid_vertices.push_back(flat_abcd_texcords[i + 2]);

        Grid_indices.push_back(glm::ivec3(Grid_vertices.size() / 3 - 3, Grid_vertices.size() / 3 - 2, Grid_vertices.size() / 3 - 1));
    }
}

void Grid::render()
{
    glBindVertexArray(Grid_VAO);
    glDrawElements(GL_TRIANGLES, (m_xPointCount - 1) * (m_yPointCount - 1) * 2 * 3, GL_UNSIGNED_INT, nullptr);
}

void Grid::renderGridLines()
{
    glBindVertexArray(Grid_VAO);
    glDrawElements(GL_LINES, (m_xPointCount - 1) * (m_yPointCount - 1) * 2 * 3, GL_UNSIGNED_INT, nullptr);
}

void Grid::renderGridPoints(float pointSize)
{
    glPointSize(pointSize);
    glBindVertexArray(Grid_VAO);
    glDrawElements(GL_POINTS, (m_xPointCount - 1) * (m_yPointCount - 1) * 2 * 3, GL_UNSIGNED_INT, nullptr);
}