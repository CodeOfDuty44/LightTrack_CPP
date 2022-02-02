#include "utils.h"

std::string readEngine(std::string const& path)
{
    std::string buffer;
    std::ifstream stream(path.c_str(), std::ios::binary);

    if (stream)
    {
        stream >> std::noskipws;
        copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), std::back_inserter(buffer));
    }

    return buffer;
}


void readPGMImage(const std::string& fileName,  uint8_t *buffer, int inH, int inW)
{
	std::cout << "[readPGMImage] Read file " << fileName << std::endl;
	std::ifstream infile(fileName, std::ios::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
	std::string magic, h, w, max;
	infile >> magic >> h >> w >> max;
    std::cout << "[readPGMImage] magic=" << magic << " h=" << h << " w=" << w << " max=" << max << std::endl;
	infile.seekg(1, infile.cur);
	infile.read(reinterpret_cast<char*>(buffer), inH*inW*3);
}

void HWC_to_CHW(uint8_t * src, uint8_t * dst, int h, int w, int chnls)
{
  //src in HWC format
  //dst in CHW format
  for( int i=0 ; i<h ; i++ )
  {
    for( int j=0 ; j<w ; j++ )
    {
      for( int c=0 ; c<chnls ; c++ )
      {
        dst[ (h*w*c) + ( i*w + j ) ] = src[ i*(w*chnls) + j*chnls + c ];
      }
    }
  }
}