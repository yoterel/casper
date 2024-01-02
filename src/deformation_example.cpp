#include "timer.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"
#include "grid.h"
#include "moving_least_squares.h"
// #include "shader_s.h"
#include "shader.h"
#include "point_cloud.h"
#include "helpers.h"
#include <iostream>
#ifdef _DEBUG
  #undef _DEBUG
  #include <Python.h>
  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
  #include <numpy/ndarrayobject.h>
  #define _DEBUG
#else
  #include <Python.h>
  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
  #include <numpy/ndarrayobject.h>
#endif
#include <filesystem>
namespace fs = std::filesystem;
#include "cnpy.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
// shader data
Shader *DrawTextureShader;
Shader *SingleColorShader;
Shader *vColorShader;

// texture data
unsigned int texture;

// Grid
Grid NormalGrid;
Grid DeformedGrid;
#define X_POINT_COUNT 41
#define Y_POINT_COUNT 41
#define X_SPACING 0.05
#define Y_SPACING 0.05
int obj = 2; // system state
bool objChanged = true;

// control points are hard coded
std::vector<cv::Point2f> ControlPointsP = {cv::Point2f(-1, -1), cv::Point2f(-1, 1), cv::Point2f(1, 1), cv::Point2f(1, -1), cv::Point2f(0.51, 0.0), cv::Point2f(-0.51, -0.0)};
std::vector<cv::Point2f> ControlPointsQ = {cv::Point2f(-1, -1), cv::Point2f(-1, 1), cv::Point2f(1, 1), cv::Point2f(1, -1), cv::Point2f(0.6, 0.0), cv::Point2f(-0.6, -0.0)};
// std::vector<cv::Point2f> ControlPointsP = { cv::Point2f(0.51, 0.0),cv::Point2f(-0.51, -0.0) };
// std::vector<cv::Point2f> ControlPointsQ = { cv::Point2f(0.4, 0.0),cv::Point2f(-0.4, -0.0)  };

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

// GL window settings
const unsigned int SCR_WIDTH = 720;
const unsigned int SCR_HEIGHT = 540;

void LoadTexture()
{
  // load and create a texture
  // -------------------------
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  // set the texture wrapping parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // set texture filtering parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // load image, create texture and generate mipmaps
  int width, height, nrChannels;
  unsigned char *data = stbi_load(std::string("../../resource/hand_render.png").c_str(), &width, &height, &nrChannels, 0);
  if (data)
  {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    // glGenerateMipmap(GL_TEXTURE_2D);
  }
  else
  {
    std::cout << "Failed to load texture" << std::endl;
  }
  stbi_image_free(data);
  // -------------------------
}

void GridProcess()
{
  // InitGrid
  NormalGrid.InitGrid(X_POINT_COUNT, Y_POINT_COUNT, X_SPACING, Y_SPACING);

  // InitDeformedGrid
  cv::Mat v = DeformedGrid.AssembleM(X_POINT_COUNT, Y_POINT_COUNT, X_SPACING, Y_SPACING);
  cv::Mat p = cv::Mat::zeros(2, ControlPointsP.size(), CV_32F);
  cv::Mat q = cv::Mat::zeros(2, ControlPointsQ.size(), CV_32F);
  // initializing p points for fish eye image
  for (int i = 0; i < ControlPointsP.size(); i++)
  {
    p.at<float>(0, i) = (ControlPointsP.at(i)).x;
    p.at<float>(1, i) = (ControlPointsP.at(i)).y;
  }
  // initializing q points for fish eye image
  for (int i = 0; i < ControlPointsQ.size(); i++)
  {
    q.at<float>(0, i) = (ControlPointsQ.at(i)).x;
    q.at<float>(1, i) = (ControlPointsQ.at(i)).y;
  }

  double a = 2.0;
  // Precompute
  cv::Mat w = MLSprecomputeWeights(p, v, a);
  cv::Mat fv;            // coordinates of keypoints
  cv::Mat A;             // for affine Deformation
  std::vector<_typeA> tA; // for similarity Deformation
  typeRigid mlsd;    // for rigid Deformation
  Timer t1;
  printf("%d \n", obj);
  t1.start();
  switch (obj)
  {
  case 0:
    A = MLSprecomputeAffine(p, v, w);
    fv = MLSPointsTransformAffine(w, A, q);
    break;

  case 1:
    tA = MLSprecomputeSimilar(p, v, w);
    fv = MLSPointsTransformSimilar(w, tA, q);
    break;

  case 2:
    mlsd = MLSprecomputeRigid(p, v, w);
    fv = MLSPointsTransformRigid(w, mlsd, q);
    break;
  default:
    break;
  }
  t1.stop();
  std::cout << "MLS Time: " << t1.getElapsedTimeInMilliSec() << std::endl;
  objChanged = false;
  if (obj < 3)
    DeformedGrid.InitDeformedGrid(X_POINT_COUNT, Y_POINT_COUNT, X_SPACING, Y_SPACING, fv);
}


std::vector<glm::vec2> mp_predict(int timestamp)
{
    // cv::Mat image;
    cv::Mat image = cv::imread("C:/src/augmented_hands/debug/ss/sg0o.0_raw_cam.png");
    // cv::resize(image1, image, cv::Size(512, 512));
    npy_intp dimensions[3] = {image.rows, image.cols, image.channels()};
    PyObject* myModule = PyImport_ImportModule("predict");
    if(!myModule)
      {
          std::cout<<"Import module failed!";
          PyErr_Print();
          return std::vector<glm::vec2>();
          // exit(1);
      }
    PyObject* predict_single = PyObject_GetAttrString(myModule,(char*)"predict_single");
    if(!predict_single)
      {
          std::cout<<"Import function failed!";
          PyErr_Print();
          return std::vector<glm::vec2>();
          // exit(1);
      }
    // warmup
    // PyObject* warmobj = PyArray_SimpleNewFromData(image.dims + 1, (npy_intp*)&dimensions, NPY_UINT8, image.data);
    // PyObject* res = PyObject_CallFunction(predict_single, "(O, i)", warmobj, 0);

    PyObject* image_object = PyArray_SimpleNewFromData(image.dims + 1, (npy_intp*)&dimensions, NPY_UINT8, image.data);
    // PyObject_CallFunction(myprint, "O", image_object);
    // PyObject* myResult = PyObject_CallFunction(iden, "O", image_object);
    PyObject* myResult = PyObject_CallFunction(predict_single, "O", image_object);
    // PyObject* myResult = PyObject_CallFunction(myprofile, "O", image_object);
    PyArrayObject* myNumpyArray = reinterpret_cast<PyArrayObject*>( myResult );
    glm::vec2* data = (glm::vec2*)PyArray_DATA(myNumpyArray);
    std::vector<glm::vec2> data_vec(data, data + PyArray_SIZE(myNumpyArray) / 2);
    // for (int j = 0; j < data_vec.size(); j+=2)
    // {
    //   std::cout << data_vec[j] << data_vec[j+1] << std::endl;
    // }
    // npy_intp* arrDims = PyArray_SHAPE( myNumpyArray );
    // int nDims = PyArray_NDIM( myNumpyArray ); // number of dimensions
      // std:: cout << nDims << std::endl;
      // for (int i = 0; i < nDims; i++)
      // {
      //   std::cout << arrDims[i] << std::endl;
      // }
      
      // cv::Mat python_result = cv::Mat(image.rows, image.cols, CV_8UC3, PyArray_DATA(myNumpyArray));
      // cv::imshow("result", python_result);
      // cv::waitKey(0);
      // double* array_pointer = reinterpret_cast<double*>( PyArray_DATA( your_numpy_array ) );
      // Py_XDECREF(myModule);
      // Py_XDECREF(myObject);
      // Py_XDECREF(myFunction);
      // Py_XDECREF(myResult);
    return data_vec;
}

void pytest()
{
    Timer tpython;
    // PySys_SetArgv(argc, (wchar_t**) argv);
    // PyObject *sys_path = PySys_GetObject("path");
    // PyList_Append(sys_path, PyUnicode_FromString("."));
    // PyRun_SimpleString("import sys");
    // PyRun_SimpleString("print(sys.path)");
    /* high level API */
    // PyRun_SimpleString("from time import time,ctime\n"
    //                      "print('Today is', ctime(time()))\n");
    /* low level API */
    // PyObject* myModule = PyImport_ImportModule("mytest");
    // PyObject* myFunction = PyObject_GetAttrString(myModule,(char*)"myabs");
    // PyObject* myResult = PyObject_CallFunction(myFunction, "d", 2.0);
    // double result = PyFloat_AsDouble(myResult);
    // std::cout << result << std::endl;
    /* numpy API */
    cv::Mat image;
    cv::Mat image1 = cv::imread("C:/src/augmented_hands/debug/ss/sg0o.0_raw_cam.png");
    cv::resize(image1, image, cv::Size(512, 512));
    npy_intp dimensions[3] = {image.rows, image.cols, image.channels()};
    PyObject* myModule = PyImport_ImportModule("predict");
    if(!myModule)
      {
          std::cout<<"Import module failed!";
          PyErr_Print();
          return;
          // exit(1);
      }
    // PyObject* init_detector = PyObject_GetAttrString(myModule,(char*)"init_detector");
    // PyObject* detector = PyObject_CallFunction(init_detector, NULL);
    // PyObject* myprint = PyObject_GetAttrString(myModule,(char*)"myprint");
    // PyObject* iden = PyObject_GetAttrString(myModule,(char*)"iden");
    // PyObject* myprofile = PyObject_GetAttrString(myModule,(char*)"myprofile");
    PyObject* predict_single = PyObject_GetAttrString(myModule,(char*)"predict_single");
    if(!predict_single)
      {
          std::cout<<"Import function failed!";
          PyErr_Print();
          return;
          // exit(1);
      }
    // warmup
    PyObject* warmobj = PyArray_SimpleNewFromData(image.dims + 1, (npy_intp*)&dimensions, NPY_UINT8, image.data);
    PyObject* res = PyObject_CallFunction(predict_single, "(O, i)", warmobj, 0);
    for (int i = 1; i < 500; i++)
    {
      tpython.start();
      PyObject* image_object = PyArray_SimpleNewFromData(image.dims + 1, (npy_intp*)&dimensions, NPY_UINT8, image.data);
      // PyObject_CallFunction(myprint, "O", image_object);
      // PyObject* myResult = PyObject_CallFunction(iden, "O", image_object);
      PyObject* myResult = PyObject_CallFunction(predict_single, "(O, i)", image_object, i);
      // PyObject* myResult = PyObject_CallFunction(myprofile, "O", image_object);
      PyArrayObject* myNumpyArray = reinterpret_cast<PyArrayObject*>( myResult );
      float* data = (float*)PyArray_DATA(myNumpyArray);
      std::vector<glm::vec2> data_vec(data, data + PyArray_SIZE(myNumpyArray));
      tpython.stop();
      // for (int j = 0; j < data_vec.size(); j+=2)
      // {
      //   std::cout << data_vec[j] << data_vec[j+1] << std::endl;
      // }
      npy_intp* arrDims = PyArray_SHAPE( myNumpyArray );
      int nDims = PyArray_NDIM( myNumpyArray ); // number of dimensions
      // std:: cout << nDims << std::endl;
      // for (int i = 0; i < nDims; i++)
      // {
      //   std::cout << arrDims[i] << std::endl;
      // }
      
      // cv::Mat python_result = cv::Mat(image.rows, image.cols, CV_8UC3, PyArray_DATA(myNumpyArray));
      // cv::imshow("result", python_result);
      // cv::waitKey(0);
      // double* array_pointer = reinterpret_cast<double*>( PyArray_DATA( your_numpy_array ) );
      // Py_XDECREF(myModule);
      // Py_XDECREF(myObject);
      // Py_XDECREF(myFunction);
      // Py_XDECREF(myResult);
    }
    std::cout << "Python Time: " << tpython.averageLapInMilliSec() << std::endl;
    // return data_vec;
}

bool loadkeypoints(std::vector<glm::vec2> &result)
{
    const fs::path keypoints_path("../../resource/hand.npy");
    cnpy::NpyArray kepoints_npy;
    if (fs::exists(keypoints_path))
    {
        result.clear();
        kepoints_npy = cnpy::npy_load(keypoints_path.string());
        std::vector<float> extract = kepoints_npy.as_vec<float>();
        for (int i = 0; i < extract.size(); i += 2)
        {
            result.push_back(glm::vec2(extract[i], extract[i + 1]));
        }
        return true;
    }
    return false;
}

int main(int argc, char *argv[])
{
  // glfw: initialize and configure
  // ------------------------------
  std::vector<glm::vec2> leap_glm;
  loadkeypoints(leap_glm);
  std::vector<cv::Point2f> keypoints = Helpers::glm2opencv(leap_glm);
  Py_Initialize();
  import_array();
  std::vector<glm::vec2> mp_glm = mp_predict(0);
  std::vector<cv::Point2f> destination = Helpers::glm2opencv(mp_glm);
  Py_Finalize();
  ControlPointsP.clear();
  ControlPointsQ.clear();
  ControlPointsP.push_back(keypoints[1]);
  ControlPointsP.push_back(keypoints[2]);
  ControlPointsP.push_back(keypoints[5]);
  ControlPointsP.push_back(keypoints[10]);
  ControlPointsP.push_back(keypoints[11]);
  ControlPointsP.push_back(keypoints[18]);
  ControlPointsP.push_back(keypoints[19]);
  ControlPointsP.push_back(keypoints[26]);
  ControlPointsP.push_back(keypoints[27]);
  ControlPointsP.push_back(keypoints[34]);
  ControlPointsP.push_back(keypoints[35]);
  ControlPointsP.push_back(keypoints[9]);
  ControlPointsP.push_back(keypoints[17]);
  ControlPointsP.push_back(keypoints[25]);
  ControlPointsP.push_back(keypoints[33]);
  ControlPointsP.push_back(keypoints[41]);

  // ControlPointsP.push_back(keypoints[7]);
  // ControlPointsP.push_back(keypoints[15]);
  // ControlPointsP.push_back(keypoints[23]);
  // ControlPointsP.push_back(keypoints[31]);
  // ControlPointsP.push_back(keypoints[39]);
  //
  ControlPointsQ.push_back(keypoints[1]);
  ControlPointsQ.push_back(keypoints[2]);
  ControlPointsQ.push_back(keypoints[5]);
  ControlPointsQ.push_back(keypoints[10]);
  ControlPointsQ.push_back(keypoints[11]);
  ControlPointsQ.push_back(keypoints[18]);
  ControlPointsQ.push_back(keypoints[19]);
  ControlPointsQ.push_back(keypoints[26]);
  ControlPointsQ.push_back(keypoints[27]);
  ControlPointsQ.push_back(keypoints[34]);
  ControlPointsQ.push_back(keypoints[35]);
  ControlPointsQ.push_back(destination[4]);
  ControlPointsQ.push_back(destination[8]);
  ControlPointsQ.push_back(destination[12]);
  ControlPointsQ.push_back(destination[16]);
  ControlPointsQ.push_back(destination[20]); 
  // ControlPointsQ.push_back(destination[3]);
  // ControlPointsQ.push_back(destination[7]);
  // ControlPointsQ.push_back(destination[11]);
  // ControlPointsQ.push_back(destination[15]);
  // ControlPointsQ.push_back(destination[19]);
  std::vector<glm::vec2> ControlPointsP_glm = Helpers::opencv2glm(ControlPointsP);
  std::vector<glm::vec2> ControlPointsQ_glm = Helpers::opencv2glm(ControlPointsQ);
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // glfw window creation
  // --------------------
  GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
  if (window == NULL)
  {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  // --------------------

  // glad: load all OpenGL function pointers
  // ---------------------------------------
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }
  // ---------------------------------------

  // build and compile our shader program
  // ------------------------------------
  DrawTextureShader = new Shader("../../src/shaders/4.1.texture.vs", "../../src/shaders/4.1.texture.fs");
  SingleColorShader = new Shader("../../src/shaders/default.vs", "../../src/shaders/default.fs");
  vColorShader = new Shader("../../src/shaders/color_by_vertex.vs", "../../src/shaders/color_by_vertex.fs");
  // ------------------------------------

  // load and create a texture
  // -------------------------
  LoadTexture();
  // -------------------------

  // render loop
  // -----------
  while (!glfwWindowShouldClose(window))
  {
    
    // Init the Grid and Get Deformed Grid
    // -------------------------
    if (objChanged)
      GridProcess();
    // -------------------------

    // input
    // -----
    processInput(window);

    // render
    // ------
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // bind Texture
    glBindTexture(GL_TEXTURE_2D, texture);

    // blend
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // render container
    DrawTextureShader->use();

    if (obj < 3)
    {
      // Render Image from 200 quad
      glBindVertexArray(DeformedGrid.Grid_VAO);
      glDrawElements(GL_TRIANGLES, DeformedGrid.Grid_indices.size() * 3, GL_UNSIGNED_INT, nullptr);
      // RenderGrid();
      // DeformedGrid.RenderGrid(SingleColorShader);
    }
    else
    {
      // Render Image from 200 quad
      glBindVertexArray(NormalGrid.Grid_VAO);
      glDrawElements(GL_TRIANGLES, NormalGrid.Grid_indices.size() * 3, GL_UNSIGNED_INT, nullptr);
      // RenderGrid();
      // NormalGrid.RenderGrid(SingleColorShader);
    }
    std::vector<glm::vec3> screen_verts_color_red = {{1.0f, 0.0f, 0.0f}};
    std::vector<glm::vec3> screen_verts_color_green = {{0.0f, 1.0f, 0.0f}};
    PointCloud cloud_src(ControlPointsP_glm, screen_verts_color_red);
    PointCloud cloud_dst(ControlPointsQ_glm, screen_verts_color_green);
    vColorShader->use();
    vColorShader->setMat4("MVP", glm::mat4(1.0f));
    cloud_src.render();
    cloud_dst.render();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();
  return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
  {
    obj = 0;
    objChanged = true;
  }
  if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
  {
    obj = 1;
    objChanged = true;
  }
  if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
  {
    obj = 2;
    objChanged = true;
  }
  if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS)
  {
    obj = 3;
  }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
  // make sure the viewport matches the new window dimensions; note that width and
  // height will be significantly larger than specified on retina displays.
  glViewport(0, 0, width, height);
}
