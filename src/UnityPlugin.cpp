
// #ifdef NATIVECPPLIBRARY_EXPORTS
// #define NATIVECPPLIBRARY_API __declspec(dllexport)
// #else
// #define NATIVECPPLIBRARY_API __declspec(dllimport)
// # endif

// ------------------------------------------------------------------------
// Plugin itself


// Link following functions C-style (required for plugins)
// extern "C"
// {

// // The functions we will call from Unity.
// //
// NATIVECPPLIBRARY_API const char*  PrintHello(){
// 	return "Hello";
// }

// NATIVECPPLIBRARY_API int PrintANumber(){
// 	return 666;
// }

// NATIVECPPLIBRARY_API int AddTwoIntegers(int a, int b) {
// 	return a + b;
// }

// NATIVECPPLIBRARY_API float AddTwoFloats(float a, float b) {
// 	return a + b;
// }

// } // end of export C block

