#include <vision.h>
#include <base_module.h>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <numeric>
#include "trt_utils.h"

std::vector<std::string> loadImageList(const std::string filename)
{
    assert(nutils::fileExists(filename));
    std::vector<std::string> imagePaths;

    FILE* f = fopen(filename.c_str(), "r");
    if (!f)
    {
        std::cout << "failed to open " << filename;
        assert(0);
    }

    //buffer with 512 bytes (more than enough)
    char str[512];
    while (fgets(str, 512, f) != NULL)
    {
        for (int i = 0; str[i] != '\0'; ++i)
        {
            if (str[i] == '\n')
            {
                str[i] = '\0';
                break;
            }
        }
        imagePaths.push_back(str);
    }
    fclose(f);
    return imagePaths;
}


std::string getFolderPath(std::string filePath)
{
  int length = filePath.size();
  for(int i=length-1; i>0; --i){
    if (filePath.compare(i,1,"/") == 0){
        length = i+1;
        break;
    }
  }
  return filePath.substr (0, length);
}


std::string getModelName(std::string filePath)
{
  int start = 0;
  int end = filePath.size()-1;

  for(int i=end; i>0; --i){
    if (filePath.compare(i,1,".") == 0){
        end = i;
        break;
    }
  }

  for(int i=end; i>0; --i){
    if (filePath.compare(i,1,"/") == 0){
        start = i+1;
        break;
    }
  }

  return filePath.substr(start, end-start);
}
