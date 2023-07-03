#ifdef NATIVECPPLIBRARY_EXPORTS
#define NATIVECPPLIBRARY_API __declspec(dllexport)
#else
#define NATIVECPPLIBRARY_API __declspec(dllimport)
# endif
extern "C" {
    class NATIVECPPLIBRARY_API UnityPlugin {
        public:
            UnityPlugin();
        // TODO: add your methods here.
    };
    NATIVECPPLIBRARY_API int nNativeCppLibrary;
    NATIVECPPLIBRARY_API int fnNativeCppLibrary(void);
    NATIVECPPLIBRARY_API int displayNumber();
    NATIVECPPLIBRARY_API int getRandom();
    NATIVECPPLIBRARY_API int displaySum();
}