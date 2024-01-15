#ifndef GRID_H
#define GRID_H
#include <glm/glm.hpp>
#include <vector>
#include <opencv2/opencv.hpp>
#include <glad/glad.h>

class Grid
{
public:
	Grid(const GLuint xPointCount, const GLuint yPointCount, const GLfloat xSpacing, const GLfloat ySpacing);
	~Grid();

	std::vector<glm::vec3> Grid_vertices;
	std::vector<glm::ivec3> Grid_indices;
	unsigned int Grid_VBO, Grid_VAO, Grid_EBO;

	// Get the space coord from input idx
	GLfloat *ComputePointCoordinates(GLuint pointIndex);
	void ComputePointCoordinate(int pointIndex, float pt[3]);

	// render all grid points for calculation

	void initGLBuffers();

	void updateGLBuffers();

	// Normal Grid
	void constructGrid();

	// Deformed Grid
	void constructDeformedGrid(cv::Mat fv);
	void constructDeformedGridSmooth(cv::Mat fv, int smooth_window);

	// Renders the actual Grid
	void render();
	void renderGridLines();
	void renderGridPoints(float pointSize = 5.0f);
	cv::Mat getM() { return M; };

private:
	cv::Mat AssembleM();
	GLuint m_xPointCount;
	GLuint m_yPointCount;
	GLfloat m_xSpacing;
	GLfloat m_ySpacing;
	cv::Mat M;
	std::vector<std::vector<glm::vec3>> flat_abcd_vec;
};

#endif // GRID_H